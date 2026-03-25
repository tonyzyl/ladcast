"""
Smoke tests for FourierLongitudeShift and LearnablePhaseGamma.

Tests:
  1. Roundtrip (forward + inverse = identity)
  2. Zero phase = identity
  3. Gradient flow through shift and phase
  4. Energy preservation (Parseval)
  5. LearnablePhaseGamma identity-biased init
  6. 5D wrapper consistency
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.models.routeB_fourier_shift import (
    FourierLongitudeShift,
    LearnablePhaseGamma,
    apply_fourier_shift_5d,
    invert_fourier_shift_5d,
)


def test_roundtrip():
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30)
    phi = torch.rand(4) * 2 * torch.pi

    z_shifted = shifter.forward_shift(z, phi)
    z_back = shifter.inverse_shift(z_shifted, phi)

    error = (z - z_back).abs().max().item()
    assert error < 1e-4, f"Roundtrip error too large: {error}"
    print(f"[PASS] Roundtrip test (max error={error:.2e})")


def test_zero_phase():
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30)
    phi = torch.zeros(4)

    z_shifted = shifter.forward_shift(z, phi)
    error = (z - z_shifted).abs().max().item()
    assert error < 1e-6, f"Zero-phase error: {error}"
    print(f"[PASS] Zero-phase test (max error={error:.2e})")


def test_gradient_flow():
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30, requires_grad=True)
    phi = torch.rand(4, requires_grad=True) * 2 * torch.pi

    z_shifted = shifter.forward_shift(z, phi)
    loss = z_shifted.sum()
    loss.backward()

    assert z.grad is not None, "No gradient for z"
    assert phi.grad is not None, "No gradient for phi"
    assert not torch.isnan(z.grad).any(), "NaN in z gradient"
    assert not torch.isnan(phi.grad).any(), "NaN in phi gradient"
    print("[PASS] Gradient flow test")


def test_energy_preservation():
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30)
    phi = torch.rand(4) * 2 * torch.pi

    orig_energy = z.pow(2).sum(dim=-1)
    shifted_energy = shifter.forward_shift(z, phi).pow(2).sum(dim=-1)

    rel_error = ((orig_energy - shifted_energy).abs() / (orig_energy.abs() + 1e-8)).max().item()
    assert rel_error < 0.01, f"Energy not preserved: relative error={rel_error}"
    print(f"[PASS] Energy preservation test (max relative error={rel_error:.4f})")


def test_gamma_identity_init():
    gamma = LearnablePhaseGamma(input_channels=84)
    z = torch.randn(4, 84, 15, 30)

    with torch.no_grad():
        delta = gamma(z)

    max_angle = delta.abs().max().item()
    assert max_angle < 0.01, f"Initial gamma output too large: {max_angle}"
    print(f"[PASS] Identity-biased init test (max initial angle={max_angle:.4f})")


def test_5d_wrappers():
    z = torch.randn(4, 1, 84, 15, 30)
    phi = torch.rand(4) * 2 * torch.pi

    z_shifted = apply_fourier_shift_5d(z, phi)
    z_back = invert_fourier_shift_5d(z_shifted, phi)

    assert z_shifted.shape == z.shape, f"Shape mismatch: {z_shifted.shape} vs {z.shape}"
    error = (z - z_back).abs().max().item()
    assert error < 1e-4, f"5D roundtrip error too large: {error}"
    print(f"[PASS] 5D wrapper test (max error={error:.2e})")


def test_gamma_gradient():
    gamma = LearnablePhaseGamma(input_channels=84)
    z = torch.randn(4, 84, 15, 30)

    delta = gamma(z)
    loss = delta.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gamma.parameters())
    assert has_grad, "No gradient flowing through gamma network"
    print("[PASS] Gamma gradient flow test")


if __name__ == "__main__":
    test_roundtrip()
    test_zero_phase()
    test_gradient_flow()
    test_energy_preservation()
    test_gamma_identity_init()
    test_5d_wrappers()
    test_gamma_gradient()
    print("\n=== All Fourier shift smoke tests passed ===")
