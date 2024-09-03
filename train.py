"""Train model."""

from __future__ import annotations

import copy
import datetime
import glob
import logging
import os
import warnings

import lpips
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from datasets import Dataset, Image
from omegaconf import OmegaConf
from PIL import Image as PILImage
from PIL import ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image

import models

# Utils


def get_logger(
    name: str,
    filename: str | None = None,
    mode: str = 'a',
    format: str = '%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s',
    auxiliary_handlers: list | None = None,
) -> logging.Logger:
    """Create logger."""
    logger = logging.getLogger(name)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(format)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if auxiliary_handlers:
        for handler in auxiliary_handlers:
            logger.addHandler(handler)

    return logger


def now_string(format: str = '%Y%m%d%H%M%S'):
    return datetime.datetime.now().strftime(format)


def cycle(iterable):
    while True:
        for data in iterable:
            yield data


def setup_pillow(load_truncated: bool = False, max_image_pixels: int | None = PILImage.MAX_IMAGE_PIXELS):
    ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated
    PILImage.MAX_IMAGE_PIXELS = max_image_pixels


# Dataset


def _is_image_file(path: str) -> bool:
    ext = {os.path.splitext(os.path.basename(path))[-1].lower()}
    return ext.issubset({'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'})


def create_dataset(
    image_folder,
    low_resolution=64,
    high_resolution=256,
    interpolation=3,
    hflip=0.5,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    resize_scale: tuple[int, int] | None = None,
):
    paths = glob.glob(os.path.join(image_folder, '**', '*'), recursive=True)
    image_paths = list(filter(_is_image_file, paths))

    dataset_dict = dict(image=image_paths)
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.sort('image')
    dataset = dataset.cast_column('image', Image(mode='RGB'))

    convert = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    resize_crop = (
        T.RandomResizedCrop(
            (int(high_resolution * 1.1), int(high_resolution * 1.1)), scale=resize_scale, ratio=[1.0, 1.0]
        )
        if resize_scale is not None
        else nn.Identity()
    )
    lr_resize = T.Resize((low_resolution, low_resolution), interpolation=interpolation)
    hr_resize = T.Resize((high_resolution, high_resolution), interpolation=interpolation)
    hflip = T.RandomHorizontalFlip(hflip)
    normalize = T.Normalize(mean, std)

    def transform_samples(batch):
        images = batch.pop('image')
        images = [resize_crop(image) for image in images]
        images = [hflip(image) for image in images]
        batch['lr'] = [normalize(lr_resize(convert(image))) for image in images]
        batch['hr'] = [normalize(hr_resize(convert(image))) for image in images]
        return batch

    dataset = dataset.with_transform(transform_samples)
    return dataset


# Loss


def gan_hinge_real_loss(logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for real images."""
    return F.relu(1.0 - logits).mean()


def gan_hinge_fake_loss(logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for fake images."""
    return F.relu(1.0 + logits).mean()


def gan_hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    return gan_hinge_real_loss(real_logits) + gan_hinge_fake_loss(fake_logits)


def gan_hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator."""
    return -fake_logits.mean()


def gan_ns_real_loss(logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating loss for real images."""
    return F.softplus(-logits).mean()


def gan_ns_fake_loss(logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating loss for fake images."""
    return F.softplus(logits).mean()


def gan_ns_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating loss for discriminator."""
    return gan_ns_real_loss(real_logits) + gan_ns_fake_loss(fake_logits)


def gan_ns_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating loss for generator."""
    return gan_ns_real_loss(fake_logits)


@torch.cuda.amp.autocast(enabled=False)
def calc_grad(outputs: torch.Tensor, inputs: torch.Tensor, scaler=None) -> torch.Tensor:
    """Calculate gradients wrt model input with torch native AMP support."""
    if isinstance(scaler, torch.cuda.amp.GradScaler):
        outputs = scaler.scale(outputs)
    ones = torch.ones(outputs.size(), device=outputs.device)
    gradients = torch.autograd.grad(
        outputs=outputs, inputs=inputs, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    if isinstance(scaler, torch.cuda.amp.GradScaler):
        gradients = gradients / scaler.get_scale()
    return gradients


@torch.no_grad()
def update_ema(
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    decay: float = 0.999,
    copy_buffers: bool = False,
) -> None:
    """Update exponential moving avg."""

    model.eval()
    param_ema = dict(model_ema.named_parameters())
    param = dict(model.named_parameters())
    for key in param_ema:
        param_ema[key].data.mul_(decay).add_(param[key].data, alpha=(1 - decay))
    if copy_buffers:
        buffer_ema = dict(model_ema.named_buffers())
        buffer = dict(model.named_buffers())
        for key in buffer_ema:
            buffer_ema[key].data.copy_(buffer[key].data)
    model.train()


def main(config_file: str = 'config/config.yaml'):
    # Load config.
    config = OmegaConf.load(config_file)

    distributed_training = torch.cuda.device_count() > 1
    if distributed_training and dist.is_torchelastic_launched():
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        if distributed_training:
            warnings.warn('The process was not lauched by torchrun. Falling back to single process.', stacklevel=1)
        rank = 0
        world_size = 1

    # Folder.
    run_folder = os.path.join(
        config.folder, '.'.join([config.name, now_string() if config.tag == 'date' else config.tag])
    )
    if rank == 0:
        os.makedirs(run_folder, exist_ok=True)

    # Logger
    if rank == 0:
        logger = get_logger(config.name, os.path.join(run_folder, 'log.log'))
    else:
        # only log to std.
        logger = get_logger(config.name + f' [rank {rank}]')

    # Save config file.
    if rank == 0:
        OmegaConf.save(config, os.path.join(run_folder, 'config.yaml'))

    # Device
    device = torch.device(config.device, rank)

    # Dataset.
    dataset = create_dataset(
        image_folder=config.data.image_folder,
        low_resolution=config.data.resolution // config.data.scale_factor,
        high_resolution=config.data.resolution,
        interpolation=config.data.interpolation,
        hflip=config.data.hflip,
        mean=OmegaConf.to_object(config.data.mean),
        std=OmegaConf.to_object(config.data.std),
        resize_scale=OmegaConf.to_object(config.data.resize_scale),
    )
    logger.info(f'\n{dataset}')
    if distributed_training:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=config.loader.shuffle, drop_last=config.loader.drop_last
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.loader.batch_size,
            sampler=sampler,
            drop_last=config.loader.drop_last,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
        )
    else:
        dataloader = DataLoader(dataset, **config.loader)
    dataloader = cycle(dataloader)

    # Models.
    G_orig = models.GigaGANUpsampler(**OmegaConf.to_object(config.g))
    if rank == 0:
        G_orig.save_config(os.path.join(run_folder, 'model-config.json'))
    logger.info(f'G\n{G_orig}')
    if rank == 0:
        G_ema = copy.deepcopy(G_orig)
        G_ema.requires_grad_(False)
        G_ema.eval()
    D = models.Discriminator(**OmegaConf.to_object(config.d))
    logger.info(f'D\n{D}')

    G_orig.to(device)
    D.to(device)
    if rank == 0:
        G_ema.to(device)

    if distributed_training:
        G = DDP(G_orig, device_ids=[rank], broadcast_buffers=False)
        D = DDP(D, device_ids=[rank], broadcast_buffers=False)
    else:
        G = G_orig

    # Optimizers
    g_optim = torch.optim.AdamW(G.parameters(), **OmegaConf.to_object(config.g_optim))
    logger.info(f'G optim\n{g_optim}')
    d_optim = torch.optim.AdamW(D.parameters(), **OmegaConf.to_object(config.d_optim))
    logger.info(f'D optim\n{d_optim}')

    # Criterion.
    d_loss_fn, g_loss_fn = (
        (gan_hinge_d_loss, gan_hinge_g_loss) if config.loss.gan_type == 'hinge' else (gan_ns_d_loss, gan_ns_g_loss)
    )
    lpips_fn = lpips.LPIPS(net='vgg')
    lpips_fn.to(device)

    # Training variables.
    assert config.train.batch_size % (config.loader.batch_size * world_size) == 0
    grad_accum_steps = int(config.train.batch_size // (config.loader.batch_size * world_size))
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    batches_done = 0

    logger.info(
        (
            f'Gradient accumulation steps: {grad_accum_steps} '
            f'({config.train.batch_size} // ({config.loader.batch_size} * {world_size}))'
        )
    )

    # Main loop.
    while batches_done < config.train.iterations:
        g_optim.zero_grad()
        d_optim.zero_grad()

        G.requires_grad_(False)
        D.requires_grad_(True)

        batches = []

        loss_totals = {'D': 0, 'G': 0, 'Dgan': 0, 'Ggan': 0, 'GP': 0, 'LPIPS': 0}  # for logging

        # Update D.
        for _ in range(grad_accum_steps):
            batch = next(dataloader)
            lr = batch.get('lr').to(device)
            hr = batch.get('hr').to(device)
            hr = hr.clone().requires_grad_(True)

            z = torch.randn(lr.size(0), config.g.z_dim, device=device)

            # keep images for g update
            batches.append((batch, z.detach().clone()))

            with torch.cuda.amp.autocast(enabled=config.amp):
                # G(x)
                with torch.no_grad():
                    fake = G(lr, z)

                # D(y)
                real_logits = D(hr)
                # D(G(x))
                fake_logits = D(fake)

                # Discriminator losses.
                d_gan_loss = d_loss_fn(real_logits, fake_logits) * config.loss.gan_lambda
                gp_loss = torch.tensor([0], device=device)
                if batches_done % config.loss.gp_every == 0:
                    gradients = calc_grad(real_logits, hr, scaler=grad_scaler)
                    gradients = gradients.reshape(gradients.size(0), -1)
                    gp_loss = gradients.norm(2, dim=1).pow(2).mean() / 2.0
                    gp_loss = gp_loss * config.loss.gp_lambda
                d_loss = d_gan_loss + gp_loss
                d_loss = d_loss / grad_accum_steps

                # for logging
                loss_totals['D'] += d_loss.item()
                loss_totals['Dgan'] += d_gan_loss.item() / grad_accum_steps
                loss_totals['GP'] += gp_loss.item() / grad_accum_steps

            grad_scaler.scale(d_loss).backward()

        for param in D.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        grad_scaler.step(d_optim)

        G.requires_grad_(True)
        D.requires_grad_(False)

        # Update G.
        lrs, fakes = [], []
        for batch, z in batches:
            lr = batch.get('lr').to(device)
            hr = batch.get('hr').to(device)
            with torch.cuda.amp.autocast(enabled=config.amp):
                # G(x)
                fake = G(lr, z)

                # D(G(x))
                fake_logits = D(fake)

                # Generator Losses.
                g_gan_loss = g_loss_fn(fake_logits) * config.loss.gan_lambda
                lpips_loss = lpips_fn(fake, hr).mean() * config.loss.lpips_lambda
                g_loss = g_gan_loss + lpips_loss
                g_loss = g_loss / grad_accum_steps

                # for logging
                loss_totals['G'] += g_loss.item()
                loss_totals['Ggan'] += g_gan_loss.item() / grad_accum_steps
                loss_totals['LPIPS'] += lpips_loss.item() / grad_accum_steps
                lrs.append(lr)
                fakes.append(fake.detach())

            grad_scaler.scale(g_loss).backward()

        for param in G.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        grad_scaler.step(g_optim)
        grad_scaler.update()

        # Update EMA parameters.
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            update_ema(G_orig, G_ema, config.train.ema_beta, copy_buffers=False)

        batches_done += 1

        # Logging, model saving.

        # Log losses.
        if (
            batches_done == 1
            or batches_done % config.log_interval == 0
            or (batches_done <= 100 and batches_done % 5 == 0)
        ):
            message_parts = [f'{batches_done: 6} / {config.train.iterations}']
            message_parts.extend([f'{key}:{value: 10.5f}' for key, value in loss_totals.items()])

            if torch.cuda.is_available():
                _, global_total = torch.cuda.mem_get_info()
                local_usage = torch.cuda.memory_reserved() / global_total * 100
                message_parts.append(f'VRAM_used(%) {local_usage:.1f}')

            logger.info(' | '.join(message_parts))

        if rank == 0:
            # Save model and sample images.
            if batches_done % config.save_every == 0:
                kbatches = f'{batches_done/1000:.2f}'

                # state_dict.
                torch.save(G_ema.state_dict(), os.path.join(run_folder, f'{kbatches}-state.pt'))

                # sample images
                test_lrs, test_fakes = [], []
                with torch.inference_mode():
                    for batch, z in batches:
                        lr = batch.get('lr').to(device)
                        fake = G_ema(lr, z)
                        test_lrs.append(lr)
                        test_fakes.append(fake)
                test_lr = torch.cat(test_lrs, dim=0)
                test_fake = torch.cat(test_fakes, dim=0)
                test_lr = F.interpolate(test_lr, hr.size()[-2:], mode='nearest')
                save_image(
                    torch.cat([test_lr, test_fake]),
                    os.path.join(run_folder, f'snapshot-{kbatches}.png'),
                    normalize=True,
                    value_range=(-1, 1),
                    pad_value=255,
                )

            # Save training (in/out)put images.
            if batches_done % config.running_every == 0 or batches_done == 1:
                kbatches = f'{batches_done/1000:.2f}'
                lr = torch.cat(lrs, dim=0)
                fake = torch.cat(fakes, dim=0)
                lr = F.interpolate(lr, hr.size()[-2:], mode='nearest')
                save_image(
                    torch.cat([lr, fake]),
                    os.path.join(run_folder, 'running.png'),
                    normalize=True,
                    value_range=(-1, 1),
                    pad_value=255,
                )

    if rank == 0:
        torch.save(G_ema.state_dict(), os.path.join(run_folder, 'final_model.pt'))


if __name__ == '__main__':
    setup_pillow(load_truncated=True, max_image_pixels=None)
    main()
