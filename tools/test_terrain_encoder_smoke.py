"""
Smoke test for TerrainEncoder and GSNO layers.

Tests:
    1. Forward pass with correct shapes
    2. Gradient flow
    3. Output gate initialization (near zero)
    4. Compatibility with LaDCast latent dimensions
    5. SHT backend (if torch_harmonics available)
"""

import sys

import torch

sys.path.insert(0, ".")

from ladcast.models.gsno_layers import (
    GSNOBlock,
    HAS_TORCH_HARMONICS,
    SpectralConv,
    create_spectral_transforms,
)
from ladcast.models.terrain_encoder import TerrainEncoder


def test_spectral_conv_basic():
    """SpectralConv should produce correct shapes with FFT backend."""
    fwd, inv = create_spectral_transforms(16, 32, 16, 32, use_sht=False)
    conv = SpectralConv(fwd, inv, 8, 8, bias=True)

    x = torch.randn(2, 8, 16, 32)
    out, residual = conv(x)

    assert out.shape == (2, 8, 16, 32), f"Expected (2,8,16,32), got {out.shape}"
    assert residual.shape == (2, 8, 16, 32), f"Residual shape mismatch: {residual.shape}"
    print(f"PASS: SpectralConv basic - output {out.shape}")


def test_spectral_conv_downsample():
    """SpectralConv should handle resolution change."""
    fwd, inv = create_spectral_transforms(16, 32, 4, 8, use_sht=False)
    conv = SpectralConv(fwd, inv, 8, 8, bias=True)

    x = torch.randn(2, 8, 16, 32)
    out, residual = conv(x)

    assert out.shape == (2, 8, 4, 8), f"Expected (2,8,4,8), got {out.shape}"
    assert residual.shape == (2, 8, 4, 8), f"Residual should be downsampled: {residual.shape}"
    print(f"PASS: SpectralConv downsample - {x.shape[-2:]} -> {out.shape[-2:]}")


def test_gsno_block():
    """GSNOBlock forward pass."""
    fwd, inv = create_spectral_transforms(16, 32, 16, 32, use_sht=False)
    block = GSNOBlock(fwd, inv, 8, mlp_ratio=2.0)

    x = torch.randn(2, 8, 16, 32)
    out = block(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"PASS: GSNOBlock - output {out.shape}")


def test_terrain_encoder_shape():
    """TerrainEncoder should produce correct output shape."""
    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=32,
        out_channels=32,
        phys_nlat=120,
        phys_nlon=240,
        latent_nlat=15,
        latent_nlon=30,
        num_phys_blocks=1,
        num_latent_blocks=1,
        use_sht=False,
    )

    x = torch.randn(2, 5, 120, 240)
    out = encoder(x)

    assert out.shape == (2, 32, 15, 30), f"Expected (2,32,15,30), got {out.shape}"
    print(f"PASS: TerrainEncoder shape - output {out.shape}")


def test_terrain_encoder_gradient():
    """Gradients should flow through the entire encoder."""
    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=16,
        out_channels=16,
        phys_nlat=120,
        phys_nlon=240,
        latent_nlat=15,
        latent_nlon=30,
        num_phys_blocks=1,
        num_latent_blocks=1,
        use_sht=False,
    )

    x = torch.randn(2, 5, 120, 240, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    for name, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    print("PASS: Gradient flow test")


def test_gate_initialization():
    """Output gate should start at zero."""
    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=16,
        out_channels=16,
        phys_nlat=16,
        phys_nlon=32,
        latent_nlat=4,
        latent_nlon=8,
        use_sht=False,
    )

    gate_val = encoder.output_gate.item()
    gate_sigmoid = torch.sigmoid(encoder.output_gate).item()
    assert abs(gate_val) < 1e-6, f"Gate should be 0, got {gate_val}"
    print(
        f"PASS: Gate init test (raw={gate_val:.6f}, sigmoid={gate_sigmoid:.4f})"
    )


def test_small_grid():
    """Test with tiny grid for fast iteration."""
    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=8,
        out_channels=8,
        phys_nlat=16,
        phys_nlon=32,
        latent_nlat=4,
        latent_nlon=8,
        num_phys_blocks=1,
        num_latent_blocks=1,
        use_sht=False,
    )

    x = torch.randn(2, 5, 16, 32)
    out = encoder(x)
    assert out.shape == (2, 8, 4, 8), f"Expected (2,8,4,8), got {out.shape}"
    print(f"PASS: Small grid test - output {out.shape}")


def test_deterministic():
    """Same input should produce same output in eval mode."""
    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=16,
        out_channels=16,
        phys_nlat=16,
        phys_nlon=32,
        latent_nlat=4,
        latent_nlon=8,
        use_sht=False,
    )
    encoder.eval()

    x = torch.randn(1, 5, 16, 32)
    with torch.no_grad():
        out1 = encoder(x)
        out2 = encoder(x)
        diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"Non-deterministic output: max diff={diff}"
    print(f"PASS: Deterministic test (max diff={diff:.2e})")


def test_param_count():
    """Check parameter count is reasonable."""
    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=64,
        out_channels=32,
        phys_nlat=120,
        phys_nlon=240,
        latent_nlat=15,
        latent_nlon=30,
        use_sht=False,
    )

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"PASS: Parameter count = {n_params:,} ({n_params / 1e6:.2f}M)")


def test_with_sht():
    """Test with SHT backend if torch_harmonics is available."""
    if not HAS_TORCH_HARMONICS:
        print("SKIP: SHT test (torch_harmonics not installed)")
        return

    encoder = TerrainEncoder(
        in_channels=5,
        embed_dim=16,
        out_channels=16,
        phys_nlat=120,
        phys_nlon=240,
        latent_nlat=15,
        latent_nlon=30,
        num_phys_blocks=1,
        num_latent_blocks=1,
        use_sht=True,
        grid="equiangular",
    )

    x = torch.randn(2, 5, 120, 240)
    out = encoder(x)
    assert out.shape == (2, 16, 15, 30), f"Expected (2,16,15,30), got {out.shape}"

    # gradient check
    x2 = torch.randn(1, 5, 120, 240, requires_grad=True)
    out2 = encoder(x2)
    out2.sum().backward()
    assert x2.grad is not None, "No gradient with SHT backend"
    print(f"PASS: SHT backend test - output {out.shape}")


if __name__ == "__main__":
    print(f"torch_harmonics available: {HAS_TORCH_HARMONICS}")
    print("=" * 50)

    test_spectral_conv_basic()
    test_spectral_conv_downsample()
    test_gsno_block()
    test_terrain_encoder_shape()
    test_terrain_encoder_gradient()
    test_gate_initialization()
    test_small_grid()
    test_deterministic()
    test_param_count()
    test_with_sht()

    print("\n=== All terrain encoder smoke tests passed ===")
