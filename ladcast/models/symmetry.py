from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SymmetryConfig:
    enabled: bool = False
    group: str = "lon_roll"
    prob: float = 0.0
    max_shift: int = 0


def sample_lon_roll_shifts(
    batch_size: int,
    max_shift: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if max_shift <= 0:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    return torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(batch_size,),
        device=device,
        generator=generator,
    )


def apply_lon_roll_5d(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Apply per-sample longitude roll on (B, C, T, H, W)."""
    out = x.clone()
    for idx, shift in enumerate(shifts.tolist()):
        if shift != 0:
            out[idx] = torch.roll(out[idx], shifts=shift, dims=-1)
    return out


def invert_lon_roll_5d(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    return apply_lon_roll_5d(x, -shifts)