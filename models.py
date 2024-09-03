"""GigaGAN"""

import json
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from stylegan2_ops.ops import bias_act, conv2d_resample, fma, upfirdn2d


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulated_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    selector: torch.Tensor = None,
    noise: torch.Tensor = None,
    up: int = 1,
    down: int = 1,
    padding: int = 0,
    resample_filter: torch.Tensor | None = None,
    demodulate: bool = True,
    flip_weight: bool = True,
    fused_modconv: bool = True,
) -> torch.Tensor:
    """Modulated Conv2d with adaptive kernel selection.

    Args:
        x (torch.Tensor): data
        weight (torch.Tensor): weight. [OIkk] or [NOIkk]
        styles (torch.Tensor): style vectors. [BI]
        selector (torch.Tensor, optional): kernel selectors. [BN]. Default: None.
        noise (torch.Tensor, optional): noise. Default: None.
        up (int, optional): up scale. Default: 1.
        down (int, optional): down scale. Default: 1.
        padding (int, optional): padding. Default: 0.
        resample_filter (torch.Tensor | None, optional): filter. Default: None.
        demodulate (bool, optional): apply demod? Default: True.
        flip_weight (bool, optional): flip weight? Default: True.
        fused_modconv (bool, optional): use fused modconv? Default: True.

    Returns:
        torch.Tensor: convolved data
    """
    B = x.size(0)

    # we automatically switch this by looking at the kernel shape.
    adaptive_kernel = False
    if weight.ndim == 5 and weight.size(0) > 1:
        assert selector is not None, '"selector" must be an Tensor.'
        N, _, Ci, kh, kw = weight.shape
        adaptive_kernel = True
        # adaptive kernel selection always uses grouped convolutions.
        fused_modconv = True
    else:
        N = 0
        _, Ci, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1 / np.sqrt(Ci * kh * kw) / weight.norm(float('inf'), dim=[-1, -2, -3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Weighted avg.
    if adaptive_kernel:
        weight = weight.unsqueeze(0)  # [1,N,Co,Ci,k,k]
        selector = selector.reshape(B, N, 1, 1, 1, 1)  # [B,N,1,1,1,1]
        weight = (weight * selector).sum(dim=1)  # [B,Co,Ci,k,k]

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) if not adaptive_kernel else weight  # [NOIkk]
        w = w * styles.reshape(B, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(B, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(B, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight,
        )
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(B, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(B, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, Ci, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=B,
        flip_weight=flip_weight,
    )
    x = x.reshape(B, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation='linear',
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        self._repr = f'in_features={in_features}, out_features={out_features}'
        self.activation = activation
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return self._repr


class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation='linear',
        up=1,
        down=1,
        resample_filter=(1, 3, 3, 1),
        conv_clamp=None,
        channels_last=False,
        trainable=True,
    ):
        super().__init__()
        self._repr = (
            f'in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, '
            f'activation={activation}, up={up}, down={down}'
        )

        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = self.up == 1  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return self._repr


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=1)
        return normed * self.scale * self.gamma


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self._repr = f'dim={dim}, dim_head={dim_head}, heads={heads}'

        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.to_q = Conv2dLayer(dim, dim_inner, 1, bias=False)
        self.to_k = Conv2dLayer(dim, dim_inner, 1, bias=False)
        self.to_v = Conv2dLayer(dim, dim_inner, 1, bias=False)

        self.to_out = Conv2dLayer(dim_inner, dim, 1, bias=False)

    def forward(self, fmap):
        # fmap = self.norm(fmap)

        x, y = fmap.shape[-2:]
        h = self.heads

        q, k, v = self.to_q(fmap), self.to_k(fmap), self.to_v(fmap)
        q, k, v = (rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=self.heads) for t in (q, k, v))

        # using pytorch cdist leads to nans in lightweight gan training framework, at least
        AA = (q * q).sum(dim=-1)
        # BB = (k * k).sum(dim=-1)
        BB = AA
        l2dist_squared = (
            rearrange(AA, 'b i -> b i 1')
            + rearrange(BB, 'b j -> b 1 j')
            - 2 * torch.einsum('b i d, b j d -> b i j', q, k)
        )
        sim = -l2dist_squared

        # scale
        sim = sim * self.scale
        # attention
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)

        return self.to_out(out)

    def extra_repr(self):
        return self._repr


class Mlp(nn.Sequential):
    def __init__(self, dim, ratio=4.0, channel_first: bool = False):
        dim_hidden = int(dim * ratio)
        linear = partial(Conv2dLayer, kernel_size=1) if channel_first else FullyConnectedLayer
        super().__init__(linear(dim, dim_hidden), nn.GELU(), linear(dim_hidden, dim))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, mlp_ratio=4.0, residual_gain: float = 1.0, norm: bool = False):
        super().__init__()
        self.gain = residual_gain
        self.attn_norm = ChannelRMSNorm(dim) if norm else nn.Identity()
        self.attn = SelfAttention(dim, dim_head, heads)
        self.mlp_norm = ChannelRMSNorm(dim) if norm else nn.Identity()
        self.mlp = Mlp(dim, mlp_ratio, channel_first=True)

    def forward(self, x):
        x = x + self.gain * self.attn(self.attn_norm(x))
        x = x + self.gain * self.mlp(self.mlp_norm(x))
        return x


class MappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=8,
        embed_features=None,
        layer_features=None,
        activation='lrelu',
        lr_multiplier=0.01,
        w_avg_beta=0.995,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


class SynthesisLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        num_kernels=4,
        up=1,
        down=1,
        use_noise=True,
        activation='lrelu',
        resample_filter=(1, 3, 3, 1),
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self._repr = (
            f'in_channels={in_channels}, out_channels={out_channels}, w_dim={w_dim}, '
            f'kernel_size={kernel_size}, num_kernels={num_kernels}, up={up}, down={down}, activation={activation}'
        )

        self.resolution = resolution
        self.up = up
        self.down = down
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.select = FullyConnectedLayer(w_dim, num_kernels) if num_kernels > 0 else nn.Identity()

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = (
            nn.Parameter(
                torch.randn([num_kernels, out_channels, in_channels, kernel_size, kernel_size]).to(
                    memory_format=memory_format
                )
            )
            if num_kernels > 1
            else nn.Parameter(
                torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
            )
        )

        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        selector = self.select(w)
        selector = selector.softmax(-1)
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            # noise = (
            #     torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
            # )
            B, _, H, W = x.size()
            H = H * self.up
            W = W * self.up
            noise = torch.randn(B, 1, H, W, device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = self.up == 1  # slightly faster
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            selector=selector,
            noise=noise,
            up=self.up,
            down=self.down,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return self._repr


class ToRGBLayer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


class SynthesisBlock(nn.Module):
    """Modified to exclude ToRGB layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        up,
        resolution,
        img_channels,
        resample_filter=(1, 3, 3, 1),
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        **layer_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=up if in_channels != 0 else 1,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

    def forward(self, x, w_iter, batch_size=None, force_fp32=False, fused_modconv=None, **layer_kwargs):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        else:
            w0 = next(w_iter)
            x = self.conv0(x, w0, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        assert x.dtype == dtype
        return x


class Residual(nn.Module):
    def __init__(self, module: nn.Module, residual_gain: float = 1.0):
        super().__init__()
        self.module = module
        self.gain = residual_gain

        if hasattr(module, 'num_conv'):
            self.num_conv = module.num_conv
            self.num_torgb = module.num_torgb

    def forward(self, x, *args, **kwargs):
        skip = x
        x = self.module(x, *args, **kwargs)
        x = x + skip * self.gain
        return x


def _no_residual(module, *args, **kwargs):
    return module


class SynthesisStage(nn.Module):
    """Synthesis stage per resolution.
    Stacks multiple synthesis block and Transformer blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        num_blocks,
        resolution,
        img_channels,
        self_attn,
        self_attn_heads,
        self_attn_ratio=1.0,
        self_attn_mlp_ratio=4.0,
        self_attn_norm=False,
        residual_gain=0.0,
        attn_residual_gain=1.0,
        resample_filter=(1, 3, 3, 1),
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        up=2,
        **layer_kwargs,
    ):
        super().__init__()
        self.up = up
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        apply_residual = partial(Residual if residual_gain > 0 else _no_residual, residual_gain=residual_gain)

        self.blocks = nn.ModuleList()
        # skip connection is always disabled when sampling is applied for simplicity.
        # This can be modified.
        self.blocks.append(
            SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                up=up,
                resolution=resolution,
                img_channels=img_channels,
                resample_filter=resample_filter,
                use_fp16=use_fp16,
                fp16_channels_last=fp16_channels_last,
                **layer_kwargs,
            )
        )
        for _ in range(num_blocks - 1):
            self.blocks.append(
                apply_residual(
                    SynthesisBlock(
                        out_channels,
                        out_channels,
                        w_dim=w_dim,
                        up=1,
                        resolution=resolution,
                        img_channels=img_channels,
                        resample_filter=resample_filter,
                        use_fp16=use_fp16,
                        fp16_channels_last=fp16_channels_last,
                        **layer_kwargs,
                    )
                )
            )

        self.num_conv = 0
        for b in self.blocks:
            self.num_conv += b.num_conv

        self.attn = (
            nn.Sequential(
                *[
                    SelfAttentionBlock(
                        out_channels,
                        int(out_channels * self_attn_ratio) // self_attn_heads,
                        heads=self_attn_heads,
                        mlp_ratio=self_attn_mlp_ratio,
                        residual_gain=attn_residual_gain,
                        norm=self_attn_norm,
                    )
                    for _ in range(self_attn)
                ]
            )
            if self_attn > 0
            else nn.Identity()
        )

        # NOTE: Currently, only supports upscaler which doesn't use cross-attention.
        self.xattn = nn.Identity()

        self.torgb = ToRGBLayer(
            out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last
        )
        self.num_torgb = 1

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        for block in self.blocks:
            x = block(
                x, w_iter, batch_size=ws.shape[0], force_fp32=force_fp32, fused_modconv=fused_modconv, **layer_kwargs
            )

        x = self.attn(x)
        x = self.xattn(x)

        # ToRGB.
        if img is not None and self.up > 1:
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

        # assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,
        img_resolution,
        img_channels,
        num_blocks,
        self_attn_depths,
        self_attn_heads,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        **block_kwargs,
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2**i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        num_blocks_dict = {
            res: num_blocks if isinstance(num_blocks, int) else num_blocks.get(res, 1) for res in self.block_resolutions
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            self_attn = self_attn_depths.get(res, -1)
            self_attn_heads_ = self_attn_heads.get(res, -1)
            block = SynthesisStage(
                in_channels,
                out_channels,
                num_blocks=num_blocks_dict[res],
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                self_attn=self_attn,
                self_attn_heads=self_attn_heads_,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws, strict=False):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


class EncoderBlock(nn.Module):
    """UNet encoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        down,
        resolution,
        residual_gain=0.0,
        resample_filter=(1, 3, 3, 1),
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        **layer_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.num_conv = 0
        self.num_torgb = 0

        self.gain = residual_gain

        self.conv0 = SynthesisLayer(
            in_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            down=down,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            use_noise=False,  # disable noise in encoder.
            **layer_kwargs,
        )
        self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            down=1,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            use_noise=False,  # disable noise in encoder.
            **layer_kwargs,
        )
        self.num_conv += 1

    def forward(self, x, w_iter, force_fp32=False, fused_modconv=None, **layer_kwargs):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        assert x.dtype == dtype
        return x


class EncoderStage(nn.Module):
    """UNet encoder stage per resolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        num_blocks,
        resolution,
        self_attn,
        self_attn_heads=8,
        self_attn_ratio=1.0,
        self_attn_mlp_ratio=4.0,
        self_attn_norm=False,
        residual_gain=0.0,
        attn_residual_gain=1.0,
        resample_filter=(1, 3, 3, 1),
        use_fp16=False,
        fp16_channels_last=False,
        down=2,
        **layer_kwargs,
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        apply_residual = partial(Residual if residual_gain > 0 else _no_residual, residual_gain=residual_gain)

        self.blocks = nn.ModuleList()
        # skip connection is always disabled when sampling is applied for simplicity.
        # This can be modified.
        self.blocks.append(
            EncoderBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                down=down,
                resolution=resolution,
                resample_filter=resample_filter,
                use_fp16=use_fp16,
                fp16_channels_last=fp16_channels_last,
                **layer_kwargs,
            )
        )
        for _ in range(num_blocks - 1):
            self.blocks.append(
                apply_residual(
                    EncoderBlock(
                        out_channels,
                        out_channels,
                        w_dim=w_dim,
                        down=1,
                        resolution=resolution,
                        resample_filter=resample_filter,
                        use_fp16=use_fp16,
                        fp16_channels_last=fp16_channels_last,
                        **layer_kwargs,
                    )
                )
            )

        self.num_conv = 0
        for b in self.blocks:
            self.num_conv += b.num_conv

        self.attn = (
            nn.Sequential(
                *[
                    SelfAttentionBlock(
                        out_channels,
                        int(out_channels * self_attn_ratio) // self_attn_heads,
                        heads=self_attn_heads,
                        mlp_ratio=self_attn_mlp_ratio,
                        residual_gain=attn_residual_gain,
                        norm=self_attn_norm,
                    )
                    for _ in range(self_attn)
                ]
            )
            if self_attn > 0
            else nn.Identity()
        )

        self.num_torgb = 0

    def forward(self, x, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        for block in self.blocks:
            x = block(x, w_iter, force_fp32=force_fp32, fused_modconv=fused_modconv, **layer_kwargs)

        x = self.attn(x)

        assert x.dtype == dtype
        return x


class UNet(torch.nn.Module):
    def __init__(
        self,
        dim,
        w_dim,
        img_channels,
        up_dim_multi=(16, 8, 4, 2, 1),
        down_dim_multi=(2, 4, 8),
        num_blocks=5,
        self_attn_depths={16: 4, 32: 2},  # noqa: B006
        self_attn_heads={16: 8, 32: 8},  # noqa: B006
        # channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,
        img_resolution=256,
        num_fp16_res=0,
        **block_kwargs,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        _encoder_dims = [img_channels, *[dim * multi for multi in [down_dim_multi[0], *down_dim_multi]]]
        _encoder_dims = [min(channel_max, channels) for channels in _encoder_dims]
        encoder_in_out = list(zip(_encoder_dims[:-1], _encoder_dims[1:], strict=False))  # noqa: RUF007
        _decoder_dims = [_encoder_dims[-1], *[dim * multi for multi in up_dim_multi], dim * up_dim_multi[-1]]
        _decoder_dims = [min(channel_max, channels) for channels in _decoder_dims]
        decoder_in_out = list(zip(_decoder_dims[:-1], _decoder_dims[1:], strict=False))  # noqa: RUF007

        self.num_ws = 0
        res = 64
        skip_channels = []
        self.encoders = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(encoder_in_out):
            if i != 0:
                res //= 2
            use_fp16 = res >= fp16_resolution
            skip_channels.append(out_channels)
            self.encoders.append(
                EncoderStage(
                    in_channels,
                    out_channels,
                    w_dim=w_dim,
                    num_blocks=num_blocks,
                    down=2 if i != 0 else 1,
                    resolution=res,
                    self_attn=0,
                    use_fp16=use_fp16,
                    **block_kwargs,
                )
            )
            self.num_ws += self.encoders[-1].num_conv
            self.num_ws += self.encoders[-1].num_torgb

        use_fp16 = res >= fp16_resolution
        self.bottom = EncoderStage(
            out_channels,
            out_channels,
            w_dim=w_dim,
            num_blocks=num_blocks,
            down=1,
            resolution=res,
            self_attn=0,
            use_fp16=use_fp16,
        )
        self.num_ws += self.bottom.num_conv
        self.num_ws += self.bottom.num_torgb

        self.decoders = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(decoder_in_out):
            use_fp16 = res >= fp16_resolution
            in_channels += skip_channels.pop() if len(skip_channels) > 0 else 0  # noqa: PLW2901

            if i != (len(decoder_in_out) - 1):
                res *= 2

            self_attn = self_attn_depths.get(res, self_attn_depths.get(str(res), -1))
            self_attn_heads_ = self_attn_heads.get(res, self_attn_heads.get(str(res), -1))

            self.decoders.append(
                SynthesisStage(
                    in_channels,
                    out_channels,
                    w_dim=w_dim,
                    up=2 if i != (len(decoder_in_out) - 1) else 1,
                    num_blocks=num_blocks,
                    resolution=res,
                    img_channels=img_channels,
                    self_attn=self_attn,
                    self_attn_heads=self_attn_heads_,
                    use_fp16=use_fp16,
                    **block_kwargs,
                )
            )
            self.num_ws += self.decoders[-1].num_conv
            self.num_ws += self.decoders[-1].num_torgb

    def forward(self, x, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for block in self.encoders:
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
            block_ws.append(ws.narrow(1, w_idx, self.bottom.num_conv + self.bottom.num_torgb))
            w_idx += self.bottom.num_conv
            for block in self.decoders:
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        block_ws = iter(block_ws)
        skip_features = []
        for stage in self.encoders:
            x = stage(x, next(block_ws), **block_kwargs)
            skip_features.append(x)
        x = self.bottom(x, next(block_ws), **block_kwargs)
        for stage in self.decoders:
            if len(skip_features) > 0:
                x = torch.cat([x, skip_features.pop()], dim=1)
            x, img = stage(x, img, next(block_ws), **block_kwargs)

        return img


class GigaGANUpsampler(nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs=None,
        synthesis_kwargs=None,
    ):
        super().__init__()
        if mapping_kwargs is None:
            mapping_kwargs = {}
        if synthesis_kwargs is None:
            synthesis_kwargs = {}
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = UNet(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        self._args = dict(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            mapping_kwargs=mapping_kwargs,
            synthesis_kwargs=synthesis_kwargs,
        )

    def save_config(self, filename: str):
        with open(filename, 'w') as fp:
            json.dump(self._args, fp, indent=2)

    @classmethod
    def from_config(cls, filename: str, weights: str | None = None):
        with open(filename, 'r') as fp:
            config = json.load(fp)
        model = cls(**config)

        if isinstance(weights, str):
            state_dict = torch.load(weights, map_location='cpu')
            model.load_state_dict(state_dict)

        return model

    def forward(self, x, z, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(x, ws, **synthesis_kwargs)
        return img


# ----------------------------------------------------------------------------


class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        tmp_channels,
        out_channels,
        resolution,
        img_channels,
        first_layer_idx,
        architecture='resnet',
        activation='lrelu',
        resample_filter=(1, 3, 3, 1),
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        freeze_layers=0,
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        cmap_dim,
        resolution,
        img_channels,
        architecture='resnet',
        mbstd_group_size=4,
        mbstd_num_channels=1,
        activation='lrelu',
        conv_clamp=None,
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = (
            MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels)
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp
        )
        self.fc = FullyConnectedLayer(in_channels * (resolution**2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


# ----------------------------------------------------------------------------


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,
        img_resolution,
        img_channels,
        architecture='resnet',
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        conv_clamp=None,
        cmap_dim=None,
        block_kwargs=None,
        mapping_kwargs=None,
        epilogue_kwargs=None,
    ):
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}
        if mapping_kwargs is None:
            mapping_kwargs = {}
        if epilogue_kwargs is None:
            epilogue_kwargs = {}

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2**i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in [*self.block_resolutions, 4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs
        )

    def forward(self, img, c=None, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x


# ----------------------------------------------------------------------------
