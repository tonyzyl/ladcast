"""
GSNO (Green's-function Spherical Neural Operator) layers adapted for LaDCast.

Provides sphere-equivariant spectral convolution on lat-lon grids.
Supports both SHT (torch_harmonics) and FFT backends.

Reference:
    "Green's-Function Spherical Neural Operators" (ICLR 2026)
    arXiv: 2512.10723v2
"""

import math

import torch
import torch.nn as nn
import torch.fft

try:
    from torch_harmonics import RealSHT, InverseRealSHT

    HAS_TORCH_HARMONICS = True
except ImportError:
    HAS_TORCH_HARMONICS = False


# ---------------------------------------------------------------------------
# FFT-based fallback transforms (used when torch_harmonics is unavailable)
# ---------------------------------------------------------------------------


class RealFFT2(nn.Module):
    """2D real FFT wrapper matching the SHT interface."""

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):
        y = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        y = torch.cat(
            (
                y[..., : math.ceil(self.lmax / 2), : self.mmax],
                y[..., -math.floor(self.lmax / 2) :, : self.mmax],
            ),
            dim=-2,
        )
        return y


class InverseRealFFT2(nn.Module):
    """Inverse 2D real FFT wrapper matching the iSHT interface."""

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):
        return torch.fft.irfft2(
            x, dim=(-2, -1), s=(self.nlat, self.nlon), norm="ortho"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def batched_spherical_average(f):
    """Area-weighted spherical average of a field on a lat-lon grid.

    Args:
        f: (B, C, N_lat, N_lon)
    Returns:
        (B, C, 1, 1) spatial average weighted by sin(theta)
    """
    B1, B2, N_theta, N_phi = f.shape
    theta = torch.linspace(0, torch.pi, N_theta, device=f.device, dtype=f.dtype)
    d_theta = theta[1] - theta[0]
    d_phi = 2 * torch.pi / N_phi
    sin_theta = torch.sin(theta).view(1, 1, N_theta, 1)
    weights = sin_theta * d_theta * d_phi
    integral = torch.sum(f * weights, dim=(-2, -1), keepdim=True)
    return integral / (4 * torch.pi)


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth (drop path) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep_prob * mask


class PointwiseMLP(nn.Module):
    """Point-wise MLP using 1x1 convolutions."""

    def __init__(
        self,
        in_features,
        out_features=None,
        hidden_features=None,
        act_layer=nn.GELU,
        gain=1.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        nn.init.normal_(fc1.weight, std=math.sqrt(2.0 / in_features))
        nn.init.constant_(fc1.bias, 0.0)

        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=False)
        nn.init.normal_(fc2.weight, std=math.sqrt(gain / hidden_features))

        self.fwd = nn.Sequential(fc1, act_layer(), fc2)

    def forward(self, x):
        return self.fwd(x)


class SpectralConv(nn.Module):
    """Spectral convolution on the sphere (Driscoll-Healy contraction).

    Supports multi-resolution: forward_transform and inverse_transform can
    operate at different spatial resolutions for downsampling/upsampling.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        gain=2.0,
        operator_type="driscoll-healy",
        bias=False,
    ):
        super().__init__()
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax
        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
            or self.forward_transform.nlon != self.inverse_transform.nlon
        )
        self.operator_type = operator_type

        weight_shape = [in_channels, in_channels]
        weight_shape_n = [in_channels, self.modes_lat, self.modes_lon]

        if operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
            self.contract_func = "...ilm,oilm->...olm"
        elif operator_type == "driscoll-healy":
            weight_shape += [self.modes_lat]
            self.contract_func = "...ilm,oil->...olm"
        else:
            raise NotImplementedError(f"Unknown operator type: {operator_type}")

        scale = math.sqrt(gain / in_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(*weight_shape, dtype=torch.complex64)
        )
        self.weight1 = nn.Parameter(
            scale * torch.randn(*weight_shape_n, dtype=torch.complex64)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        residual = x
        b = x.shape[0]

        x_int = batched_spherical_average(x)

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)

        w1 = self.weight1.unsqueeze(0).expand(b, -1, -1, -1)
        x = x + x_int * w1
        x = torch.einsum(self.contract_func, x, self.weight)

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        return x.to(dtype), residual


class GSNOBlock(nn.Module):
    """Single GSNO block: SpectralConv + inner skip + MLP + outer skip."""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        channels,
        mlp_ratio=2.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        use_mlp=True,
    ):
        super().__init__()

        gain = 1.0  # gain_factor with act / inner_skip

        self.spectral_conv = SpectralConv(
            forward_transform,
            inverse_transform,
            channels,
            channels,
            gain=gain,
            bias=True,
        )

        self.inner_skip = nn.Conv2d(channels, channels, 1)
        nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain / channels))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_mlp:
            self.mlp = PointwiseMLP(
                channels,
                channels,
                hidden_features=int(channels * mlp_ratio),
                act_layer=act_layer,
                gain=0.5,
            )

        self.outer_skip = nn.Conv2d(channels, channels, 1)
        nn.init.normal_(self.outer_skip.weight, std=math.sqrt(0.5 / channels))

    def forward(self, x):
        x_spec, residual = self.spectral_conv(x)
        x_spec = x_spec + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x_spec = self.mlp(x_spec)

        x_spec = self.drop_path(x_spec)
        x_spec = x_spec + self.outer_skip(residual)

        return x_spec


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_spectral_transforms(
    nlat_in,
    nlon_in,
    nlat_out=None,
    nlon_out=None,
    grid="equiangular",
    use_sht=True,
):
    """Create forward/inverse spectral transform pair.

    If nlat_out/nlon_out differ from input, the transforms perform
    resolution change (spectral truncation for downsampling).
    Falls back to FFT if torch_harmonics is not available.
    """
    nlat_out = nlat_out or nlat_in
    nlon_out = nlon_out or nlon_in

    modes_lat = min(nlat_in, nlat_out) // 2
    modes_lon = min(nlon_in, nlon_out) // 2 + 1

    if use_sht and HAS_TORCH_HARMONICS:
        fwd = RealSHT(
            nlat_in, nlon_in, lmax=modes_lat, mmax=modes_lon, grid=grid
        ).float()
        inv = InverseRealSHT(
            nlat_out, nlon_out, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
        ).float()
    else:
        fwd = RealFFT2(nlat_in, nlon_in, lmax=modes_lat, mmax=modes_lon)
        inv = InverseRealFFT2(nlat_out, nlon_out, lmax=modes_lat, mmax=modes_lon)

    return fwd, inv
