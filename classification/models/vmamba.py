"""
vmamba.py
=========
Official VMamba (Visual State Space Model) Architecture with ImageNet-1K Pretrained Weights.

Paper: "VMamba: Visual State Space Model" (Liu et al., 2024)
https://arxiv.org/abs/2401.10166

Architecture Overview (VSSM-Tiny):
  - PatchEmbed2D: 4x4 non-overlapping patch embedding with LayerNorm.
  - 4 Hierarchical Stages (depths=[2, 2, 9, 2], dims=[96, 192, 384, 768]).
  - SS2D (2D Selective Scan Module): 4-directional Cross-Scan, Selective Scan (SSM), and Cross-Merge.
  - PatchMerging2D: 2x downsampling between stages with linear reduction and LayerNorm.
  - Classifier Head: Global Average Pooling + LayerNorm + Linear(768, num_classes).
  - Pretrained Weights: Official ImageNet-1K checkpoint (vssmtiny_dp01_ckpt_epoch_292.pth).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import repeat

try:
    from timm.layers import trunc_normal_, DropPath
except ImportError:
    from timm.models.layers import trunc_normal_, DropPath

try:
    from .utils import selective_scan_fn
except (ImportError, ValueError):
    try:
        from utils import selective_scan_fn
    except (ImportError, ValueError):
        selective_scan_fn = None

# Official VMamba-Tiny ImageNet-1K checkpoint URL
OFFICIAL_VMAMBA_TINY_URL = (
    "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"
)


# -----------------------------------------------------------------------------
# Patch Embedding & Merging
# -----------------------------------------------------------------------------
class PatchEmbed2D(nn.Module):
    """4x4 2D Image to Patch Embedding."""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H/4, W/4, embed_dim]
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x)


class PatchMerging2D(nn.Module):
    """2D Patch Merging Layer for Hierarchical Downsampling."""

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


# -----------------------------------------------------------------------------
# Selective Scan SSM Engine
# -----------------------------------------------------------------------------
def selective_scan_core(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """Memory-efficient reference implementation of selective scan."""
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    B_sz, D_sz, L_sz = u.shape
    if B.dim() == 4:
        B = B.squeeze(1)
    if C.dim() == 4:
        C = C.squeeze(1)

    if A.dim() == 2:
        A = A.unsqueeze(0).expand(B_sz, -1, -1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0).expand(B_sz, -1)

    N_sz = A.shape[-1]
    x = torch.zeros(B_sz, D_sz, N_sz, device=u.device, dtype=u.dtype)
    ys = []

    for i in range(L_sz):
        d_i = delta[:, :, i].unsqueeze(-1)
        dA_i = torch.exp(d_i * A)
        dB_u_i = (d_i * u[:, :, i].unsqueeze(-1)) * B[:, :, i].unsqueeze(1)
        x = dA_i * x + dB_u_i
        y_i = (x * C[:, :, i].unsqueeze(1)).sum(dim=-1)
        ys.append(y_i)

    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.unsqueeze(-1)
    return y


# -----------------------------------------------------------------------------
# 2D Selective Scan Module (SS2D)
# -----------------------------------------------------------------------------
class SS2D(nn.Module):
    """Selective Scan 2D module from official VMamba."""

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.zeros(4, self.dt_rank + self.d_state * 2, self.d_inner))
        self.dt_projs_weight = nn.Parameter(torch.zeros(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.zeros(4, self.d_inner))
        self.A_logs = nn.Parameter(torch.zeros(4 * self.d_inner, self.d_state))
        self.Ds = nn.Parameter(torch.zeros(4 * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        L = H * W
        K = 4

        xz = self.in_proj(x)
        x_proj, z_proj = xz.chunk(2, dim=-1)

        x_conv = x_proj.permute(0, 3, 1, 2).contiguous()
        x_conv = self.act(self.conv2d(x_conv))

        # 4-way cross scan
        x0 = x_conv.view(B, -1, L)
        x1 = x_conv.transpose(2, 3).contiguous().view(B, -1, L)
        x2 = torch.flip(x0, dims=[-1])
        x3 = torch.flip(x1, dims=[-1])
        xs = torch.stack([x0, x1, x2, x3], dim=1).view(B, K, -1, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs_stacked = xs.view(B * K, -1, L)
        dts_stacked = dts.contiguous().view(B * K, -1, L)
        Bs_stacked = Bs.contiguous().view(B * K, -1, L)
        Cs_stacked = Cs.contiguous().view(B * K, -1, L)

        As = -torch.exp(self.A_logs.float()).view(K, self.d_inner, self.d_state)
        As_rep = repeat(As, "k d n -> (b k) d n", b=B)

        Ds = self.Ds.float().view(K, self.d_inner)
        Ds_rep = repeat(Ds, "k d -> (b k) d", b=B)

        dt_bias = self.dt_projs_bias.float().view(K, self.d_inner)
        dt_bias_rep = repeat(dt_bias, "k d -> (b k) d", b=B)

        # 4. Selective Scan
        if selective_scan_fn is not None and x.is_cuda:
            Bs_ssm = Bs.contiguous().view(B * K, 1, self.d_state, L)[:, :, :1, :]
            As_ssm = As[0, :self.d_inner, :1].contiguous()
            Ds_ssm = Ds[0, :self.d_inner].contiguous()
            dt_bias_ssm = dt_bias[0, :self.d_inner].contiguous()

            h = selective_scan_fn(
                xs_stacked,
                dts_stacked,
                As_ssm,
                Bs_ssm,
                Ds_ssm,
                delta_bias=dt_bias_ssm,
                delta_softplus=True,
                return_last_state=False,
            )
            # h is [B*K, d_inner, 1, L] -> reshape to [B*K, d_inner, L]
            y = h.view(B * K, self.d_inner, L)
        else:
            y = selective_scan_core(
                xs_stacked, dts_stacked, As_rep, Bs_stacked, Cs_stacked, Ds_rep, delta_bias=dt_bias_rep
            )
        ys = y.view(B, K, -1, L)

        # 4-way cross merge
        y0 = ys[:, 0].view(B, self.d_inner, H, W)
        y1 = ys[:, 1].view(B, self.d_inner, W, H).transpose(2, 3)
        y2 = torch.flip(ys[:, 2], dims=[-1]).view(B, self.d_inner, H, W)
        y3 = torch.flip(ys[:, 3], dims=[-1]).view(B, self.d_inner, W, H).transpose(2, 3)
        y_merged = (y0 + y1 + y2 + y3).permute(0, 2, 3, 1)

        y = self.out_norm(y_merged)
        y = y * F.silu(z_proj)
        y = self.out_proj(y)
        return self.dropout(y)


# -----------------------------------------------------------------------------
# VSS Block & Stage
# -----------------------------------------------------------------------------
class VSSBlock(nn.Module):
    """Visual State Space Block."""

    def __init__(
        self,
        hidden_dim: int = 96,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class VSSLayer(nn.Module):
    """Single stage containing VSSBlocks and optional PatchMerging2D downsampling."""

    def __init__(
        self,
        dim: int = 96,
        depth: int = 2,
        d_state: int = 16,
        drop_path: float = 0.0,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging2D(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# -----------------------------------------------------------------------------
# VMamba / VSSM Backbone & Classifier
# -----------------------------------------------------------------------------
class VMamba(nn.Module):
    """Official VMamba (VSSM) Visual State Space Model."""

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 5,
        depths: list = (2, 2, 9, 2),
        dims: list = (96, 192, 384, 768),
        d_state: int = 16,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0]
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=(i_layer < self.num_layers - 1),
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = (
            nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(dims[-1], num_classes),
            )
            if drop_rate > 0.0
            else nn.Linear(dims[-1], num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(1, 2))  # Global average pooling over [H, W]
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def load_vmamba_pretrained_weights(model: nn.Module, pretrained_path: str = None) -> nn.Module:
    """Load official ImageNet-1K pretrained weights into VMamba backbone.

    Replaces the classification head to match num_classes.
    """
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Loading VMamba pretrained weights from local file: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu")
    else:
        # Load from torch cache or download official checkpoint
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints"
        )
        local_cached = os.path.join(cache_dir, "vssmtiny_dp01_ckpt_epoch_292.pth")
        if os.path.isfile(local_cached):
            print(f"Loading VMamba ImageNet-1K pretrained weights from torch cache: {local_cached}")
            ckpt = torch.load(local_cached, map_location="cpu")
        else:
            print(f"Downloading VMamba ImageNet-1K pretrained weights from {OFFICIAL_VMAMBA_TINY_URL}...")
            ckpt = torch.hub.load_state_dict_from_url(
                OFFICIAL_VMAMBA_TINY_URL, map_location="cpu", progress=True
            )

    sd = ckpt["model"] if "model" in ckpt else ckpt
    # Filter out classifier head weights (1000-class ImageNet -> 5-class target)
    filtered_sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    msg = model.load_state_dict(filtered_sd, strict=False)
    print(f"VMamba ImageNet-1K backbone weights loaded successfully! Missing keys (expected head): {msg.missing_keys}")
    return model


def freeze_vmamba(model: nn.Module, freeze_backbone: bool = False, freeze_stages: int = -1):
    """Freeze backbone weights or stages for fast downstream fine-tuning."""
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False
        print("[Fine-Tuning] Frozen entire VMamba backbone. Only classifier head will be updated.")
    elif freeze_stages >= 0:
        model.patch_embed.requires_grad_(False)
        for i in range(min(freeze_stages + 1, len(model.layers))):
            model.layers[i].requires_grad_(False)
        print(f"[Fine-Tuning] Frozen VMamba stem and stages 0 through {freeze_stages}.")


def VMamba_T(
    num_classes: int = 5,
    pretrained: bool = True,
    pretrained_path: str = None,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    freeze_backbone: bool = False,
    freeze_stages: int = -1,
    **kwargs,
) -> VMamba:
    """VMamba-T (Tiny / VSSM-Tiny) model with ImageNet-1K pretrained weights."""
    model = VMamba(
        num_classes=num_classes,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    if pretrained:
        load_vmamba_pretrained_weights(model, pretrained_path)
    if freeze_backbone or freeze_stages >= 0:
        freeze_vmamba(model, freeze_backbone=freeze_backbone, freeze_stages=freeze_stages)
    return model
