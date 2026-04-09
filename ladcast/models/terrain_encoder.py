"""
GSNO-based Terrain Encoder for LaDCast.

Processes static geographic fields (orography, land-sea mask) through
sphere-equivariant spectral convolutions, producing terrain features
at latent spatial resolution for conditioning the diffusion model.

Usage:
    encoder = TerrainEncoder(
        in_channels=5,        # 1 LSM + 4 orography
        embed_dim=64,
        out_channels=32,
        phys_nlat=120,        # physical grid (after south pole crop)
        phys_nlon=240,
        latent_nlat=15,       # latent grid
        latent_nlon=30,
    )

    # static_fields: (B, 5, 120, 240)
    terrain_features = encoder(static_fields)  # (B, 32, 15, 30)
"""

import math

import torch
import torch.nn as nn

from ladcast.models.gsno_layers import (
    GSNOBlock,
    SpectralConv,
    create_spectral_transforms,
)


class TerrainEncoder(nn.Module):
    """Encodes static terrain fields into latent-resolution features
    using GSNO spectral convolutions for sphere-equivariant processing.

    Architecture:
        1. Lift: 1x1 conv to embed_dim
        2. GSNO blocks at physical resolution (sphere-equivariant features)
        3. Spectral downsampling to latent resolution (SHT truncation)
        4. GSNO block at latent resolution (refinement)
        5. Project: 1x1 conv to out_channels

    The output_gate is initialized to zero so that terrain conditioning
    is gradually introduced during training (identity-biased init).
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 64,
        out_channels: int = 32,
        phys_nlat: int = 120,
        phys_nlon: int = 240,
        latent_nlat: int = 15,
        latent_nlon: int = 30,
        num_phys_blocks: int = 2,
        num_latent_blocks: int = 1,
        mlp_ratio: float = 2.0,
        use_sht: bool = True,
        grid: str = "equiangular",
    ):
        super().__init__()

        self.phys_nlat = phys_nlat
        self.phys_nlon = phys_nlon
        self.latent_nlat = latent_nlat
        self.latent_nlon = latent_nlon

        # 1. Input projection
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=True),
            nn.GELU(),
        )
        nn.init.normal_(self.encoder[0].weight, std=math.sqrt(2.0 / in_channels))
        nn.init.constant_(self.encoder[0].bias, 0.0)

        # 2. GSNO blocks at physical resolution
        fwd_phys, inv_phys = create_spectral_transforms(
            phys_nlat, phys_nlon, phys_nlat, phys_nlon,
            grid=grid, use_sht=use_sht,
        )
        self.phys_blocks = nn.ModuleList(
            [
                GSNOBlock(fwd_phys, inv_phys, embed_dim, mlp_ratio=mlp_ratio)
                for _ in range(num_phys_blocks)
            ]
        )

        # 3. Spectral downsampling: physical -> latent resolution
        fwd_down, inv_down = create_spectral_transforms(
            phys_nlat, phys_nlon, latent_nlat, latent_nlon,
            grid=grid, use_sht=use_sht,
        )
        self.downsample = SpectralConv(
            fwd_down, inv_down, embed_dim, embed_dim,
            gain=1.0, bias=True,
        )

        # Learned downsampling residual (for robustness)
        self.downsample_skip = nn.AdaptiveAvgPool2d((latent_nlat, latent_nlon))

        # 4. GSNO blocks at latent resolution
        fwd_lat, inv_lat = create_spectral_transforms(
            latent_nlat, latent_nlon, latent_nlat, latent_nlon,
            grid=grid, use_sht=use_sht,
        )
        self.latent_blocks = nn.ModuleList(
            [
                GSNOBlock(fwd_lat, inv_lat, embed_dim, mlp_ratio=mlp_ratio)
                for _ in range(num_latent_blocks)
            ]
        )

        # 5. Output projection
        self.decoder = nn.Conv2d(embed_dim, out_channels, 1, bias=False)
        nn.init.normal_(self.decoder.weight, std=math.sqrt(1.0 / embed_dim))

        # Gating: start near zero so terrain conditioning is gradually introduced
        self.output_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_static, H_phys, W_phys) static terrain fields
               e.g., (B, 5, 120, 240) for 1 LSM + 4 orography

        Returns:
            terrain_features: (B, out_channels, H_latent, W_latent)
                              e.g., (B, 32, 15, 30)
        """
        # 1. Lift
        h = self.encoder(x)

        # 2. Physical-resolution GSNO blocks
        for block in self.phys_blocks:
            h = block(h)

        # 3. Downsample to latent resolution
        h_spec, _ = self.downsample(h)
        h_skip = self.downsample_skip(h)
        h = h_spec + h_skip

        # 4. Latent-resolution refinement
        for block in self.latent_blocks:
            h = block(h)

        # 5. Project and gate
        out = self.decoder(h)
        out = out * self.output_gate.sigmoid()

        return out
