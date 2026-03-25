from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ladcast.models.symmetry import apply_lon_roll_5d, invert_lon_roll_5d, sample_lon_roll_shifts


class TinyLatentAR(nn.Module):
    """Minimal latent AR predictor: (B, Sin, C, H, W) -> (B, Sout, C, H, W)."""

    def __init__(self, in_seq: int, out_seq: int, channels: int):
        super().__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.channels = channels
        self.proj = nn.Conv2d(in_seq * channels, out_seq * channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = x.shape
        x = x.reshape(b, s * c, h, w)
        y = self.proj(x)
        return y.reshape(b, self.out_seq, c, h, w)


class CircularLonConv2d(nn.Module):
    """Conv2d with circular padding in longitude and zero padding in latitude."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="constant", value=0.0)
        return self.conv(x)



def _group_norm_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=_group_norm_groups(channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=_group_norm_groups(channels), num_channels=channels)
        self.conv1 = CircularLonConv2d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CircularLonConv2d(channels, channels, kernel_size=kernel_size, dilation=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class GridResNetBackbone(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int = 128,
            num_blocks: int = 6,
            kernel_size: int = 3,
            dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.temporal_mixer = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.stem = CircularLonConv2d(hidden_channels, hidden_channels, kernel_size=kernel_size)
        self.stem_act = nn.GELU()

        dilations = [1, 2, 1, 4, 1, 2]
        blocks = []
        for idx in range(num_blocks):
            blocks.append(
                ResBlock2d(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilations[idx % len(dilations)],
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=_group_norm_groups(hidden_channels), num_channels=hidden_channels),
            nn.GELU(),
            CircularLonConv2d(hidden_channels, hidden_channels, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_mixer(x)
        x = self.stem_act(self.stem(x))
        x = self.blocks(x)
        x = self.head(x)
        return x


class _RouteBResNetBase(nn.Module):
    def __init__(
            self,
            in_seq: int,
            out_seq: int,
            channels: int,
            hidden_channels: int = 128,
            num_blocks: int = 6,
            kernel_size: int = 3,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.channels = channels
        self.backbone = GridResNetBackbone(
            in_channels=in_seq * channels,
            out_channels=out_seq * channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def _backbone_predict(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = x.shape
        return self.backbone(x.reshape(b, s * c, h, w)).reshape(b, self.out_seq, c, h, w)


class RouteBNonSymmResNet(_RouteBResNetBase):
    """RouteB ResNet baseline using the same backbone as symm_resnet without roll symmetrisation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone_predict(x)


class RouteBSymmResNet(_RouteBResNetBase):
    """RouteB grid latent predictor with stochastic longitude symmetrisation."""

    def __init__(
            self,
            in_seq: int,
            out_seq: int,
            channels: int,
            hidden_channels: int = 128,
            num_blocks: int = 6,
            kernel_size: int = 3,
            dropout: float = 0.0,
            max_lon_shift: int = 16,
    ):
        super().__init__(
            in_seq=in_seq,
            out_seq=out_seq,
            channels=channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.max_lon_shift = int(max_lon_shift)

    def _predict_single(self, x: torch.Tensor, shifts: torch.Tensor | None) -> torch.Tensor:
        if shifts is not None:
            x = apply_lon_roll_5d(x, shifts)
        y = self._backbone_predict(x)
        if shifts is not None:
            y = invert_lon_roll_5d(y, shifts)
        return y

    def forward(
            self,
            x: torch.Tensor,
            *,
            inference_mode: str = "auto",
            num_symmetry_samples: int = 1,
            deterministic_shift: torch.Tensor | None = None,
            aggregate: str = "mean",
    ) -> torch.Tensor:
        if num_symmetry_samples <= 0:
            raise ValueError("num_symmetry_samples must be > 0")

        if deterministic_shift is not None:
            shifts = deterministic_shift.to(device=x.device, dtype=torch.long)
            return self._predict_single(x, shifts)

        if inference_mode == "auto":
            inference_mode = "train_random" if self.training else "random_single"

        if inference_mode == "deterministic":
            shifts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self._predict_single(x, shifts)

        if inference_mode in {"train_random", "random_single"}:
            shifts = None
            if self.max_lon_shift > 0:
                shifts = sample_lon_roll_shifts(x.shape[0], self.max_lon_shift, x.device)
            return self._predict_single(x, shifts)

        if inference_mode not in {"group_mean", "group_median", "group_raw"}:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")

        members = []
        for _ in range(num_symmetry_samples):
            shifts = None
            if self.max_lon_shift > 0:
                shifts = sample_lon_roll_shifts(x.shape[0], self.max_lon_shift, x.device)
            members.append(self._predict_single(x, shifts))

        raw_members = torch.stack(members, dim=0)
        if inference_mode == "group_raw" or aggregate == "none":
            return raw_members
        if inference_mode == "group_median" or aggregate == "median":
            return raw_members.median(dim=0).values
        return raw_members.mean(dim=0)


__all__ = ["TinyLatentAR", "RouteBNonSymmResNet", "RouteBSymmResNet"]
