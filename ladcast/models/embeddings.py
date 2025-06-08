import math
from datetime import datetime
from numbers import Number
from typing import List, Optional, Tuple, Union

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from einops import rearrange
from torch import nn


class SimplifiedRectangularPatchEmbed(nn.Module):
    """
    No pos embedding, intended for rotary positional embedding
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return hidden_states


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return hidden_states


class LaDCastLevelPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B,C,L,H,W) -> (B, (H*W), L*C)"""
        hidden_states = self.proj(hidden_states)
        hidden_states = rearrange(hidden_states, "B C L H W -> B (H W) (L C)")
        return hidden_states


class LaDCastRotaryPosEmbed(nn.Module):
    """
    Modified from HunyuanVideoRotaryPosEmbed
    """

    def __init__(
        self,
        rope_dim_list: List[int],
        patch_size_list: List[int],
        theta: Union[float, List[float]] = 10000,
        rope_grid_start_pos_list: Optional[
            Union[List[Optional[Number]], Tuple[Optional[Number]]]
        ] = None,
        rope_grid_end_pos_list: Optional[
            Union[List[Optional[Number]], Tuple[Optional[Number]]]
        ] = None,
    ) -> None:
        super().__init__()

        assert len(rope_dim_list) == len(patch_size_list), (
            f"RoPE dimensions {rope_dim_list} must match the patch size {patch_size_list}"
        )
        self.patch_size_list = patch_size_list
        self.rope_dim_list = rope_dim_list
        if isinstance(theta, (Number)):
            theta = [theta] * len(rope_dim_list)
        self.theta = theta
        self.rope_grid_start_pos_list = rope_grid_start_pos_list
        self.rope_grid_end_pos_list = rope_grid_end_pos_list

        if self.rope_grid_start_pos_list is not None:
            assert len(self.rope_grid_start_pos_list) == len(
                self.rope_grid_end_pos_list
            ), (
                "Length of rope_grid_start_pos_list and rope_grid_end_pos_list must be the same"
            )
            for i in range(len(self.rope_grid_start_pos_list)):
                assert (self.rope_grid_start_pos_list[i] is None) == (
                    self.rope_grid_end_pos_list[i] is None
                ), (
                    f"Element {i} of rope_grid_start_pos_list and rope_grid_end_pos_list must be both None or not None"
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_grid_start_pos_list: Optional[
            Union[List[Optional[Number]], Tuple[Optional[Number]]]
        ] = None,
        rope_grid_end_pos_list: Optional[
            Union[List[Optional[Number]], Tuple[Optional[Number]]]
        ] = None,
    ) -> torch.Tensor:
        """
        hidden_states: [B, C, T, H, W], [B, C, H, W], or [B, C, ...]
        """
        with torch.autocast("cuda", torch.float32, cache_enabled=True):
            axes_grids = []
            assert len(hidden_states.shape) - 2 == len(self.rope_dim_list), (
                f"Hidden states shall have shape [B, C, ...], where dim(...) got {len(hidden_states.shape) - 2} shall match the RoPE dimensions {self.rope_dim_list}"
            )
            rope_sizes = [
                dim_length // patch_size
                for dim_length, patch_size in zip(
                    hidden_states.shape[2:], self.patch_size_list
                )
            ]

            for i in range(len(self.rope_dim_list)):
                if rope_grid_start_pos_list is None:
                    rope_grid_start_pos = (
                        self.rope_grid_start_pos_list[i]
                        if self.rope_grid_start_pos_list[i] is not None
                        else 0
                    )
                    rope_grid_end_pos = (
                        self.rope_grid_end_pos_list[i]
                        if self.rope_grid_end_pos_list[i] is not None
                        else hidden_states.shape[i + 2] - 1
                    )
                else:
                    assert (rope_grid_start_pos_list[i] is None) == (
                        rope_grid_end_pos_list[i] is None
                    ), (
                        f"Element {i} of rope_grid_start_pos_list and rope_grid_end_pos_list must be both None or not None"
                    )
                    rope_grid_start_pos = (
                        rope_grid_start_pos_list[i]
                        if rope_grid_start_pos_list[i] is not None
                        else 0
                    )
                    rope_grid_end_pos = (
                        rope_grid_end_pos_list[i]
                        if rope_grid_end_pos_list[i] is not None
                        else hidden_states.shape[i + 2] - 1
                    )
                grid = torch.linspace(
                    rope_grid_start_pos,
                    rope_grid_end_pos,
                    steps=rope_sizes[i],
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                axes_grids.append(grid)

            grid = torch.meshgrid(*axes_grids, indexing="ij")  # [T ,H, W] or [H, W]
            grid = torch.stack(grid, dim=0)  # [3, T, H, W] or [2, H, W]

            freqs = []
            for i in range(len(self.rope_dim_list)):
                freq = get_1d_rotary_pos_embed(
                    self.rope_dim_list[i],
                    grid[i].reshape(-1),
                    self.theta[i],
                    use_real=True,
                )
                freqs.append(freq)

            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)

        return freqs_cos, freqs_sin


def get_patch_center_grid_from_num_patches(
    patch_start: float,
    patch_end: float,
    num_patches: int,
    interval_between_point: float,
    deg2rad: bool = True,
    grid_start: Optional[float] = None,
    grid_end: Optional[float] = None,
    device=None,
):
    """
    The start and end of grid shall be the start and end of the rectangular grid, patch start can be arbitrary
    position within the grid.
    num_patches: can be interpreted as number of intervals
    e.g. 240x120 (cropped south pole): lat: (start=-88..5, end=90, num_patches=15, interval_between_point=1.5)
                                       lon: (start=0, end=358.5, num_patches=30, interval_between_point=1.5)
    """
    # TODO: need to add support for augmented starting point
    grid_start = patch_start if grid_start is None else grid_start
    grid_end = patch_end if grid_end is None else grid_end

    num_points = (grid_end - grid_start) / interval_between_point + 1
    assert num_points % num_patches == 0, (
        f"Number of points {num_points} must be divisible by number of patches {num_patches}"
    )
    num_points_per_patch = num_points / num_patches
    dist_of_each_patch_center_to_its_start = (
        interval_between_point * (num_points_per_patch - 1) / 2
    )
    grid = torch.linspace(
        grid_start + dist_of_each_patch_center_to_its_start,
        grid_end - dist_of_each_patch_center_to_its_start,
        num_patches,
        device=device,
        dtype=torch.float32,
    )
    if deg2rad:
        grid = torch.deg2rad(grid)
    return grid


class LaDCastRotaryPosEmbed_from_grid(nn.Module):
    """
    Modified from HunyuanVideoRotaryPosEmbed
    """

    def __init__(
        self,
        rope_dim_list: List[int],
        patch_size_list: List[int],
        theta: Union[float, List[float]] = 10000,
    ) -> None:
        super().__init__()

        assert len(rope_dim_list) == len(patch_size_list), (
            f"RoPE dimensions {rope_dim_list} must match the patch size {patch_size_list}"
        )
        self.patch_size_list = patch_size_list
        self.rope_dim_list = rope_dim_list
        if isinstance(theta, (Number)):
            theta = [theta] * len(rope_dim_list)
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_grid_list: Union[List[List[Number]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        hidden_states: [B, C, T, H, W], [B, C, H, W], or [B, C, ...]
        """
        with torch.autocast("cuda", torch.float32, cache_enabled=True):
            axes_grids = []
            assert len(hidden_states.shape) - 2 == len(self.rope_dim_list), (
                f"Hidden states shall have shape [B, C, ...], where dim(...) got {len(hidden_states.shape) - 2} shall match the RoPE dimensions {len(self.rope_dim_list)}"
            )
            rope_sizes = [
                dim_length // patch_size
                for dim_length, patch_size in zip(
                    hidden_states.shape[2:], self.patch_size_list
                )
            ]

            for i in range(len(self.rope_dim_list)):
                assert len(rope_grid_list[i]) == rope_sizes[i], (
                    f"RoPE grid size {len(rope_grid_list[i])} must match the RoPE size {rope_sizes[i]}, at dim {i}"
                )
                if not isinstance(rope_grid_list[i], torch.Tensor):
                    grid = torch.tensor(
                        rope_grid_list[i],
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                else:
                    grid = rope_grid_list[i].to(
                        hidden_states.device, dtype=torch.float32
                    )
                axes_grids.append(grid)

            grid = torch.meshgrid(*axes_grids, indexing="ij")  # [T ,H, W] or [H, W]
            grid = torch.stack(grid, dim=0)  # [3, T, H, W] or [2, H, W]

            freqs = []
            for i in range(len(self.rope_dim_list)):
                freq = get_1d_rotary_pos_embed(
                    self.rope_dim_list[i],
                    grid[i].reshape(-1),
                    self.theta[i],
                    use_real=True,
                )
                # print('idx: ', i, 'freq: ', freq[0].shape, freq[0])
                freqs.append(freq)

            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)

        return freqs_cos, freqs_sin


def get_rotary_emb_from_surface_pressure(
    pressure_tensor: torch.Tensor,
    rope_dim: int,
    theta: int = 10000,
    pool_size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_Pa2hPa: bool = False,
):
    """
    In this scenario, the pressure_tensor is treated as the grid value as the input to the get_id_rotary_pos_embed
    Args:
        pressure_tensor: (B, 1, H, W)
    """
    bs = pressure_tensor.shape[0]
    if pool_size is not None:
        pressure_tensor = nn.functional.avg_pool2d(
            pressure_tensor, kernel_size=pool_size
        )
    if scale_Pa2hPa:
        pressure_tensor = pressure_tensor * 0.01  # Pa to hPa
    # print('pressure_tensor: ', pressure_tensor.reshape(-1)[0])
    freq_cos, freq_sin = get_1d_rotary_pos_embed(
        dim=rope_dim, pos=pressure_tensor.reshape(-1), theta=theta, use_real=True
    )  # (B*S, D)
    freq_cos = freq_cos.unflatten(0, (bs, -1))  # (B, S, D)
    freq_sin = freq_sin.unflatten(0, (bs, -1))  # (B, S, D)
    return freq_cos, freq_sin


def assemble_rotary_embedding(
    spatial_rot_emb,
    atm_var_level_rot_emb,
    sur_var_ground_level_rot_emb,
    sur_var_sea_level_rot_emb,
):
    """
    Assemble the rotary positional embedding.

    Args:
        spatial_rot_emb: Tensor of shape (N_spatial, C_spatial)
        atm_var_level_rot_emb: Tensor of shape (N_atm, C_level) e.g., (13, C_level)
        sur_var_ground_level_rot_emb: Tensor of shape (B, N_spatial, C_level)
        sur_var_sea_level_rot_emb: Tensor of shape (1, C_level)

    Returns:
        final_emb: Tensor of shape (B, N_spatial, (N_atm+2) * (C_spatial+C_level))
    """
    B = sur_var_ground_level_rot_emb.shape[0]
    N_spatial, C_spatial = spatial_rot_emb.shape
    N_atm, C_level = atm_var_level_rot_emb.shape
    C_emb = C_spatial + C_level  # per-segment embedding size

    # ---- Assemble atm segments ----
    # Expand the spatial embedding to (B, N_atm, N_spatial, C_spatial)
    spatial_atm = (
        spatial_rot_emb.unsqueeze(0).unsqueeze(0).expand(B, N_atm, N_spatial, C_spatial)
    )
    # Expand atm-level embedding from (N_atm, C_level) to (B, N_atm, N_spatial, C_level)
    atm_var_level_rot_emb = (
        atm_var_level_rot_emb.unsqueeze(0)
        .unsqueeze(2)
        .expand(B, N_atm, N_spatial, C_level)
    )
    atm_segments = torch.cat(
        [atm_var_level_rot_emb, spatial_atm], dim=-1
    )  # (B, N_atm, N_spatial, C_emb)
    atm_segments = atm_segments.transpose(1, 2).reshape(B, N_spatial, N_atm * C_emb)

    # ---- Assemble sur ground segment ----
    # Expand spatial embedding to (B, N_spatial, C_spatial)
    spatial_common = spatial_rot_emb.unsqueeze(0).expand(B, N_spatial, C_spatial)
    sur_ground_seg = torch.cat(
        [sur_var_ground_level_rot_emb, spatial_common], dim=-1
    )  # (B, N_spatial, C_emb)

    # ---- Assemble sur sea segment ----
    # Expand sur sea level embedding from (1, C_level) to (B, N_spatial, C_level)
    sur_var_sea_level_rot_emb = sur_var_sea_level_rot_emb.unsqueeze(0).expand(
        B, N_spatial, C_level
    )
    sur_sea_seg = torch.cat(
        [sur_var_sea_level_rot_emb, spatial_common], dim=-1
    )  # (B, N_spatial, C_emb)

    # Concatenate along the feature dimension: (B, N_spatial, (N_atm+2) * C_emb)
    return torch.cat([atm_segments, sur_ground_seg, sur_sea_seg], dim=-1)


#################################################################################
#                  Timestamp Conversion and Year Progress Functions                   #
#################################################################################


def convert_timestamp_to_int(timestamp: str) -> int:
    """Convert timestamp string in format 'YYYY-MM-DDThh' to integer YYYYMMDDHH."""
    clean_timestamp = timestamp.replace("-", "").replace("T", "")
    return int(clean_timestamp)


def convert_int_to_datetime(timestamp_int: int) -> datetime:
    """Convert integer in format YYYYMMDDHH to datetime object."""
    timestamp_str = str(timestamp_int)

    # Extract components
    year = int(timestamp_str[0:4])
    month = int(timestamp_str[4:6])
    day = int(timestamp_str[6:8])
    hour = int(timestamp_str[8:10])

    # Create datetime object
    return datetime(year, month, day, hour)


def compute_year_progress(dt: datetime) -> float:
    year_start = datetime(dt.year, 1, 1)
    year_end = datetime(dt.year + 1, 1, 1)
    total_seconds_in_year = (year_end - year_start).total_seconds()
    seconds_elapsed = (dt - year_start).total_seconds()
    return seconds_elapsed / total_seconds_in_year


def interpolate_time_elapsed(cur_ts: float, start_ts: float, end_ts: float) -> float:
    return start_ts + (end_ts - start_ts) * cur_ts


def timestamp_tensor_to_time_elapsed(timestamp_tensor: torch.Tensor) -> torch.Tensor:
    """timestamp_tensor: (B,) tensor of timestamps in int format."""
    return_tensor = torch.empty_like(
        timestamp_tensor, dtype=torch.float32, device=timestamp_tensor.device
    )
    return_tensor.fill_(float("nan"))
    for i, ts in enumerate(timestamp_tensor):
        dt = convert_int_to_datetime(ts.item())
        year_progress = compute_year_progress(dt)
        return_tensor[i] = year_progress
    return return_tensor


def get_year_sincos_embedding(
    timestamp_tensor: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
    freq_scale: Optional[int] = 1,
) -> torch.Tensor:
    """
    Create sinusoidal embeddings for year progress using fixed frequencies and phase inputs.

    Args:
        timestamp_tensor: (B,) tensor of timestamps in int format (YYYYMMDDHH)
        embedding_dim: Dimension of the embedding
        max_period: Controls the magnitude scaling of the sinusoidal functions

    Returns:
        (B, embedding_dim) tensor of embeddings
    """
    # Convert timestamps to normalized year progress (0 to 1)
    time_elapsed = timestamp_tensor_to_time_elapsed(timestamp_tensor)

    # Ensure embedding_dim is even for sin/cos pairs
    half_dim = embedding_dim // 2
    batch_size = timestamp_tensor.shape[0]

    # Create embedding tensor
    embedding = torch.zeros((batch_size, embedding_dim), device=timestamp_tensor.device)

    # Create fixed frequencies for different dimensions
    # These remain constant regardless of time_elapsed
    frequencies = (
        torch.arange(1, half_dim + 1, device=timestamp_tensor.device).float()
        * freq_scale
    )

    # Create magnitude scaling factors using exponential decay
    # This suppresses higher frequency components
    magnitude_scale = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, device=timestamp_tensor.device).float()
        / half_dim
    )

    # Use time_elapsed directly as phase input (scaled to 2π for full cycle)
    phase = 2 * math.pi * time_elapsed.reshape(-1, 1)  # Shape: [batch_size, 1]

    # First half: superposition of sine waves with different frequencies
    # phase * frequencies creates different phase shifts for each frequency component
    sin_args = phase * frequencies.reshape(1, -1)  # Shape: [batch_size, half_dim]
    embedding[:, :half_dim] = torch.sin(sin_args) * magnitude_scale.reshape(1, -1)

    # Second half: superposition of cosine waves with different frequencies
    embedding[:, half_dim:] = torch.cos(sin_args) * magnitude_scale.reshape(1, -1)

    return embedding
