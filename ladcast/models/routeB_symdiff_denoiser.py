from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ladcast.models.routeB_latent_ar import CircularLonConv2d, _group_norm_groups
from ladcast.models.routeB_fourier_shift import apply_fourier_shift_5d, invert_fourier_shift_5d
from ladcast.models.symmetry import apply_lon_roll_5d, invert_lon_roll_5d


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        timesteps = timesteps.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SymDiffResBlock2d(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=_group_norm_groups(channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=_group_norm_groups(channels), num_channels=channels)
        self.conv1 = CircularLonConv2d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CircularLonConv2d(channels, channels, kernel_size=kernel_size, dilation=1)
        self.time_proj = nn.Linear(time_embed_dim, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.dropout(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class RouteBSymDiffDenoiser(nn.Module):
    """Conditional epsilon denoiser for routeB latent diffusion with stochastic symmetrisation.

    Args:
        symmetry_type: "roll" (integer circular shift, legacy) or "fourier" (phase rotation).
            Determines how group_state is interpreted in forward():
            - "roll": group_state is (B,) LongTensor of integer shifts
            - "fourier": group_state is (B,) FloatTensor of phase angles in radians
    """

    def __init__(
            self,
            channels: int,
            cond_seq_len: int = 1,
            target_seq_len: int = 1,
            hidden_channels: int = 128,
            num_blocks: int = 6,
            kernel_size: int = 3,
            dropout: float = 0.0,
            time_embed_dim: int = 256,
            symmetry_type: str = "roll",
    ):
        super().__init__()
        if symmetry_type not in ("roll", "fourier"):
            raise ValueError(f"Unknown symmetry_type: {symmetry_type}")
        self.channels = channels
        self.cond_seq_len = cond_seq_len
        self.target_seq_len = target_seq_len
        self.symmetry_type = symmetry_type
        in_channels = (cond_seq_len + target_seq_len) * channels
        out_channels = target_seq_len * channels

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.stem = CircularLonConv2d(hidden_channels, hidden_channels, kernel_size=kernel_size)
        dilations = [1, 2, 1, 4, 1, 2]
        self.blocks = nn.ModuleList(
            [
                SymDiffResBlock2d(
                    hidden_channels,
                    time_embed_dim=time_embed_dim,
                    kernel_size=kernel_size,
                    dilation=dilations[idx % len(dilations)],
                    dropout=dropout,
                )
                for idx in range(num_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.GroupNorm(num_groups=_group_norm_groups(hidden_channels), num_channels=hidden_channels),
            nn.GELU(),
            CircularLonConv2d(hidden_channels, hidden_channels, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(
            self,
            cond: torch.Tensor,
            noisy_target: torch.Tensor,
            t: torch.Tensor,
            group_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond.ndim != 5 or noisy_target.ndim != 5:
            raise ValueError("cond and noisy_target must be 5D tensors")
        if cond.shape[0] != noisy_target.shape[0]:
            raise ValueError("cond and noisy_target batch sizes must match")

        if group_state is not None:
            if self.symmetry_type == "roll":
                cond = apply_lon_roll_5d(cond, group_state)
                noisy_target = apply_lon_roll_5d(noisy_target, group_state)
            else:  # fourier
                cond = apply_fourier_shift_5d(cond, group_state)
                noisy_target = apply_fourier_shift_5d(noisy_target, group_state)

        b, s_in, c, h, w = cond.shape
        _, s_out, c_out, h_out, w_out = noisy_target.shape
        if c != self.channels or c_out != self.channels:
            raise ValueError("Channel count mismatch for RouteBSymDiffDenoiser")
        if s_in != self.cond_seq_len or s_out != self.target_seq_len:
            raise ValueError("Sequence length mismatch for RouteBSymDiffDenoiser")
        if h != h_out or w != w_out:
            raise ValueError("Spatial size mismatch between cond and noisy_target")

        x = torch.cat(
            [cond.reshape(b, s_in * c, h, w), noisy_target.reshape(b, s_out * c_out, h, w_out)],
            dim=1,
        )
        t_emb = self.time_embed(t)
        x = self.input_proj(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, t_emb)
        pred_eps = self.output_head(x).reshape(b, s_out, c_out, h, w)

        if group_state is not None:
            if self.symmetry_type == "roll":
                pred_eps = invert_lon_roll_5d(pred_eps, group_state)
            else:  # fourier
                pred_eps = invert_fourier_shift_5d(pred_eps, group_state)
        return pred_eps
