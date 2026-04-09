"""
Fourier-domain phase rotation: soft symmetrisation replacing torch.roll.

Key properties:
  - Differentiable (supports gradient backprop through phase angles)
  - Preserves frequency energy spectrum (only modifies phase)
  - forward_shift + inverse_shift = identity (exact in float precision)
  - Does not depend on spatial translation equivariance

Usage:
    shifter = FourierLongitudeShift()
    z_shifted = shifter.forward_shift(z, phase_angle)
    z_back = shifter.inverse_shift(z_shifted, phase_angle)
    # z_back ≈ z (exact within float precision)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FourierLongitudeShift(nn.Module):
    """Apply phase rotation along the longitude (last) dimension in Fourier domain."""

    def forward_shift(self, z: torch.Tensor, phase_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [..., W] tensor — last dim is longitude. Supports 4D (B,C,H,W) or 5D (B,S,C,H,W).
            phase_angle: (B,) per-sample phase shift in radians [0, 2π).
        Returns:
            z_shifted: same shape as z.
        """
        W = z.shape[-1]
        Z = torch.fft.rfft(z, dim=-1)  # [..., W//2+1]
        freq = torch.arange(Z.shape[-1], device=z.device, dtype=z.dtype)

        # Broadcast phase_angle (B,) to match Z shape
        # phase_angle (B,) → (B, 1, ..., 1) then * freq (F,) → (B, 1, ..., 1, F) via broadcast
        pa = phase_angle
        for _ in range(z.ndim - 1):
            pa = pa.unsqueeze(-1)  # (B,) → (B,1) → (B,1,1) → ...
        phase = pa * freq  # broadcast: (B, 1, ..., 1) * (F,) → (B, 1, ..., F)

        rotation = torch.complex(torch.cos(phase), torch.sin(phase))
        # For even W, the Nyquist bin (last rfft coeff) must stay real for
        # Hermitian symmetry.  Rotating it introduces an imaginary part that
        # irfft silently drops, breaking the roundtrip.  Fix: don't rotate it.
        if W % 2 == 0:
            rotation[..., -1] = torch.complex(
                torch.ones_like(phase[..., -1]),
                torch.zeros_like(phase[..., -1]),
            )
        Z_shifted = Z * rotation
        return torch.fft.irfft(Z_shifted, n=W, dim=-1)

    def inverse_shift(self, z: torch.Tensor, phase_angle: torch.Tensor) -> torch.Tensor:
        """Exact inverse: apply negative phase rotation."""
        return self.forward_shift(z, -phase_angle)

    def forward(self, z: torch.Tensor, phase_angle: torch.Tensor) -> torch.Tensor:
        return self.forward_shift(z, phase_angle)


# ---------------------------------------------------------------------------
# 5D wrappers (B, S, C, H, W) — analogous to apply_lon_roll_5d
# ---------------------------------------------------------------------------

_SHARED_SHIFTER = FourierLongitudeShift()


def apply_fourier_shift_5d(x: torch.Tensor, phase_angle: torch.Tensor) -> torch.Tensor:
    """Apply per-sample Fourier phase shift on 5D tensor (B, S, C, H, W)."""
    B, S, C, H, W = x.shape
    # Reshape to (B, S*C, H, W), apply shift, reshape back
    flat = x.reshape(B, S * C, H, W)
    shifted = _SHARED_SHIFTER.forward_shift(flat, phase_angle)
    return shifted.reshape(B, S, C, H, W)


def invert_fourier_shift_5d(x: torch.Tensor, phase_angle: torch.Tensor) -> torch.Tensor:
    """Invert per-sample Fourier phase shift on 5D tensor (B, S, C, H, W)."""
    B, S, C, H, W = x.shape
    flat = x.reshape(B, S * C, H, W)
    unshifted = _SHARED_SHIFTER.inverse_shift(flat, phase_angle)
    return unshifted.reshape(B, S, C, H, W)


# ---------------------------------------------------------------------------
# Learnable gamma network for recursive symmetrisation
# ---------------------------------------------------------------------------

class LearnablePhaseGamma(nn.Module):
    """
    Learnable gamma network that outputs phase angles for Fourier symmetrisation.

    Follows SymDiff recursive symmetrisation:
      1. Sample phi_0 ~ Uniform(0, 2π)         (Haar measure)
      2. Inverse-transform input: z' = shift(z, -phi_0)
      3. Predict increment: delta = gamma_1(z', eta)
      4. Final angle: phi = phi_0 + delta

    Identity-biased initialisation: initial output ≈ 0 → gamma ≈ no transform.
    """

    def __init__(self, input_channels: int, hidden_dim: int = 256, noise_dim: int = 16):
        super().__init__()
        self.noise_dim = noise_dim

        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.noise_proj = nn.Linear(noise_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

        # Identity-biased init: output starts near zero
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, eta: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W) — already inverse-shifted by phi_0.
            eta: (B, noise_dim) — external noise; auto-sampled if None.
        Returns:
            delta_angle: (B,) increment phase angle.
        """
        if eta is None:
            eta = torch.randn(z.shape[0], self.noise_dim, device=z.device, dtype=z.dtype)
        h = self.encoder(z) + self.noise_proj(eta)
        return torch.tanh(self.head(h).squeeze(-1)) * torch.pi


__all__ = [
    "FourierLongitudeShift",
    "apply_fourier_shift_5d",
    "invert_fourier_shift_5d",
    "LearnablePhaseGamma",
]
