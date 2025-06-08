import calendar
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr


def periodic_rearrange(tensor: torch.Tensor, coords: torch.Tensor):
    """
    Rearrange a tensor based on new top-left coordinates.

    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    coords (torch.Tensor): Tensor of shape (2,) containing (x, y) coordinates.

    Returns:
    torch.Tensor: Rearranged tensor
    """
    if coords.shape != (2,):
        raise ValueError(f"Coordinates shape must be (2,), got {coords.shape}")

    # Separate the y and x shifts
    shift_y = -coords[1]  # Shifting in the height dimension (H)
    shift_x = -coords[0]  # Shifting in the width dimension (W)
    # Apply torch.roll to the tensor based on the shifts
    rearranged_tensor = torch.roll(
        tensor, shifts=(shift_y.item(), shift_x.item()), dims=(-2, -1)
    )

    return rearranged_tensor


def periodic_rearrange_batch(
    tensor: torch.Tensor, coords: Optional[torch.Tensor] = None
):
    """
    Rearrange a batch of tensors based on new top-left coordinates.

    Args:
    tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
    coords (Optional[torch.Tensor]): Tensor of shape (B, 2) containing (x, y) coordinates of the top left for each batch.
                                     If None, random coordinates will be generated.

    Returns:
    torch.Tensor: Rearranged tensor
    """
    B, C, H, W = tensor.shape

    if coords is None:
        # Generate random coordinates if not provided
        new_x = torch.randint(0, W, (B,))
        new_y = torch.randint(0, H, (B,))
        coords = torch.stack([new_x, new_y], dim=1)

    if coords.shape != (B, 2):
        raise ValueError(f"Coordinates shape must be {(B, 2)}, got {coords.shape}")

    rearranged_tensors = torch.empty_like(tensor)
    for i in range(B):
        shift_y = -coords[i, 1].item()  # Shifting in the height dimension (H)
        shift_x = -coords[i, 0].item()  # Shifting in the width dimension (W)
        rearranged_tensors[i] = torch.roll(
            tensor[i], shifts=(shift_y, shift_x), dims=(-2, -1)
        )

    return rearranged_tensors


def split_combined_dataset(merged_ds):
    """
    Convert the 69 var merged xarray back to original 69 variables xarr.
    Splits a merged xarray.Dataset containing a combined variable with surface and atmospheric variables
    into a separated Dataset with atmospheric variables having (time, level, latitude, longitude)
    and surface variables having (time, latitude, longitude).

    Parameters:
    - merged_ds: xarray.Dataset containing the combined variable 'combined_surface_and_atmospheric_variables_69'

    Returns:
    - separated_ds: xarray.Dataset with separated atmospheric and surface variables
    """
    combined_var_name = "combined_surface_and_atmospheric_variables_69"
    if combined_var_name not in merged_ds:
        raise ValueError(
            f"Combined variable '{combined_var_name}' not found in the dataset."
        )

    combined_da = merged_ds[combined_var_name]
    channel_names = merged_ds["channel"].values
    pattern = re.compile(r"^(?P<var>.+)_level_(?P<level>\d+(\.\d+)?)$")

    atmospheric_vars = []
    atmospheric_levels = []
    atmospheric_channels = []

    surface_vars = []
    surface_channels = []

    for name in channel_names:
        match = pattern.match(name)
        if match:
            var = match.group("var")
            level = int(match.group("level"))
            atmospheric_vars.append(var)
            atmospheric_levels.append(level)
            atmospheric_channels.append(name)
        else:
            surface_vars.append(name)
            surface_channels.append(name)

    unique_atm_vars = sorted(list(set(atmospheric_vars)))

    atmospheric_data_vars = {}

    for var in unique_atm_vars:
        var_channels = [
            name for name in atmospheric_channels if name.startswith(f"{var}_level_")
        ]
        if not var_channels:
            raise ValueError(f"No channels found for atmospheric variable '{var}'.")

        var_levels = [int(pattern.match(name).group("level")) for name in var_channels]

        sorted_indices = np.argsort(var_levels)
        var_levels_sorted = np.array(var_levels)[sorted_indices]
        var_channels_sorted = [var_channels[i] for i in sorted_indices]

        var_data = combined_da.sel(channel=var_channels_sorted)
        var_data = var_data.assign_coords(level=("channel", var_levels_sorted))
        var_data = var_data.swap_dims({"channel": "level"})
        var_data = var_data.drop_vars("channel")
        atmospheric_data_vars[var] = var_data

    surface_data_vars = {}
    for var in surface_vars:
        var_data = combined_da.sel(channel=var)
        var_data = var_data.drop_vars("channel")
        surface_data_vars[var] = var_data

    atmospheric_ds = xr.Dataset(atmospheric_data_vars)
    surface_ds = xr.Dataset(surface_data_vars)

    separated_ds = xr.merge([atmospheric_ds, surface_ds])

    return separated_ds


def normalize_transform_3D(sample, mean, std, target_std=1):
    # (C, T, H, W)
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=sample.device)
        std = torch.tensor(std, device=sample.device)
    return (
        (sample - mean[:, None, None, None]) / std[:, None, None, None]
    ) * target_std


def inverse_normalize_transform_3D(normalized_sample, mean, std, target_std=1):
    # (C, T, H, W)
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=normalized_sample.device)
        std = torch.tensor(std, device=normalized_sample.device)
    return (normalized_sample / target_std) * std[:, None, None, None] + mean[
        :, None, None, None
    ]


def get_transform_3D(transform, transform_args):
    """(C, T, H, W), the transform_args shall be at most 'C' channel-wise"""
    if transform == "normalize":
        mean = transform_args["mean"]
        std = transform_args["std"]
        if "target_std" in transform_args:
            target_std = transform_args["target_std"]
            return lambda x: normalize_transform_3D(x, mean, std, target_std)
        return lambda x: normalize_transform_3D(x, mean, std)
    elif transform is None:
        return lambda x: x
    else:
        raise NotImplementedError(f"Transform: {transform} not implemented.")


def get_inv_transform_3D(transform, transform_args):
    if transform == "normalize":
        mean = transform_args["mean"]
        std = transform_args["std"]
        if "target_std" in transform_args:
            target_std = transform_args["target_std"]
            return lambda x: inverse_normalize_transform_3D(x, mean, std, target_std)
        return lambda x: inverse_normalize_transform_3D(x, mean, std)
    elif transform is None:
        return lambda x: x
    else:
        raise NotImplementedError(f"Transform: {transform} not implemented.")


def precompute_mean_std(normalization_param_dict: Dict, variable_names: list):
    """
    Precompute the mean and std tensors for the given variables. variable_names shall match the order of the input.
    Returns two tensors: one for means and one for stds.
    """
    mean_list = []
    std_list = []

    for var_name in variable_names:
        if var_name in normalization_param_dict:
            norm_params = normalization_param_dict[var_name]

            if isinstance(norm_params["mean"], dict):
                # If the variable has level-based mean and std, create one tensor per level
                for level in norm_params["mean"].keys():
                    mean_list.append(norm_params["mean"][level])
                    std_list.append(norm_params["std"][level])
            else:
                # For regular variables, add mean and std directly
                mean_list.append(norm_params["mean"])
                std_list.append(norm_params["std"])
        else:
            # If no normalization info is found, use 0 mean and 1 std (no normalization)
            raise ValueError(
                f"No normalization parameters found for variable {var_name}."
            )
            # mean_list.append(0)
            # std_list.append(1)
            # Warning(f"No normalization parameters found for variable {var_name}. Using 0 mean and 1 std.")

    # Convert lists to PyTorch tensors
    mean_tensor = torch.tensor(mean_list)
    std_tensor = torch.tensor(std_list)

    return mean_tensor, std_tensor


def xarr_varname_to_tensor(
    xarr: xr.Dataset,
    variable_names: str,
    level: Optional[list[int]] = None,
    expected_static_variable_names: Optional[list[str]] = ["land_sea_mask"],
) -> Tuple[torch.Tensor, int]:
    """
    return tensor Tuple((C, T, H, W), sst_channel_idx)
    """
    list_of_tensors = []
    sst_channel_idx = None
    cur_channel_idx = 0
    level = level if level is not None else list(xarr["level"].values)
    for var_name in variable_names:
        if "time" not in xarr[var_name].dims:
            if var_name not in expected_static_variable_names:
                Warning(
                    f"Unexpected variable {var_name} does not have 'time' dimension. Skipping."
                )
            continue
        elif "level" in xarr[var_name].dims:
            # If the variable has level dimension, select the specified levels
            selected_var = (
                xarr[var_name]
                .sel(level=level)
                .transpose("level", "time", "latitude", "longitude")
            )
            selected_var = selected_var.values
            cur_channel_idx += len(level)
        else:
            if var_name == "sea_surface_temperature":
                # Special case for sea surface temperature (SST)
                sst_channel_idx = cur_channel_idx
            selected_var = (
                xarr[var_name]
                .transpose("time", "latitude", "longitude")
                .values[None, ...]
            )
            cur_channel_idx += 1
        # print(f"Processing variable: {var_name}, shape: {selected_var.shape}, current channel index: {cur_channel_idx}")
        list_of_tensors.append(selected_var)

    return torch.from_numpy(
        np.concatenate(list_of_tensors, axis=0)
    ), sst_channel_idx  # (C, T, H, W)


@torch.no_grad()
def xarr_to_tensor(
    xarr: xr.Dataset,
    variable_names: Optional[list[str]] = None,
    level: Optional[list[int]] = None,
    normalization_param_dict: Optional[Dict] = None,
    mean_tensor: Optional[torch.Tensor] = None,
    std_tensor: Optional[torch.Tensor] = None,
    add_static: bool = False,  # append land-sea mask and 4 orography at the last channel
    normalize_static: bool = True,  # normalize lsm & orography
    static_conditioning_tensor: Optional[
        torch.Tensor
    ] = None,  # if provided, will be used instead of extracting from xarr
) -> torch.Tensor:
    """
    return (C, T, H, W)
    """
    assert "time" in xarr.dims, "xarr must have 'time' dimension."
    if variable_names is None:
        variable_names = list(xarr.data_vars.keys())
    expected_static_variable_names = ["land_sea_mask"]
    if level is None:
        level = list(xarr["level"].values)

    return_tensor, sst_channel_idx = xarr_varname_to_tensor(
        xarr, variable_names, level, expected_static_variable_names
    )
    # exclude lsm if it is in the variable_names
    var_name_to_normalize = [
        var_name for var_name in variable_names if var_name != "land_sea_mask"
    ]
    if normalization_param_dict is not None:
        mean_tensor, std_tensor = precompute_mean_std(
            normalization_param_dict, var_name_to_normalize
        )  # do not normalize lsm

    if mean_tensor is not None:
        return_tensor = normalize_transform_3D(return_tensor, mean_tensor, std_tensor)

    if sst_channel_idx is not None:
        # see weather_dataset.py
        if mean_tensor is not None:
            nan_mask = torch.isnan(return_tensor[sst_channel_idx])  # (B, H, W)
            return_tensor[sst_channel_idx][nan_mask] = -2

    if add_static:
        if static_conditioning_tensor is not None:
            # If lsm_tensor is provided, use it directly
            static_conditioning_tensor = static_conditioning_tensor.to(
                return_tensor.dtype
            )
        else:
            lsm_tensor = torch.from_numpy(
                xarr["land_sea_mask"].transpose("latitude", "longitude").values
            ).to(return_tensor.dtype)
            orography_tensor = torch.from_numpy(
                xarr[
                    [
                        "standard_deviation_of_orography",
                        "angle_of_sub_gridscale_orography",
                        "anisotropy_of_sub_gridscale_orography",
                        "slope_of_sub_gridscale_orography",
                    ]
                ]
                .transpose("latitude", "longitude")
                .to_dataarray()
                .values
            ).to(return_tensor.dtype)  # (4, H, W)
            print(lsm_tensor.shape, orography_tensor.shape)
            static_conditioning_tensor = torch.cat(
                [lsm_tensor.unsqueeze(0), orography_tensor], dim=0
            )  # (5, H, W)
            if normalize_static:
                static_mean_tensor = static_conditioning_tensor.mean(
                    dim=(1, 2), keepdim=True
                ).to(return_tensor.device)  # (C, 1, 1)
                static_std_tensor = static_conditioning_tensor.std(
                    dim=(1, 2), keepdim=True
                ).to(return_tensor.device)
                static_conditioning_tensor = (
                    (static_conditioning_tensor - static_mean_tensor)
                    / static_std_tensor
                ).to(return_tensor.device)
        # expand to (C, T, H, W)
        static_conditioning_tensor = static_conditioning_tensor.unsqueeze(1).expand(
            -1, return_tensor.shape[1], -1, -1
        )
        return_tensor = torch.cat(
            [return_tensor, static_conditioning_tensor], dim=0
        )  # (C+5, T, H, W)

    return return_tensor


@torch.no_grad()
def tensor_to_xarr(
    input_tensor: torch.Tensor,
    xarr_meta: xr.Dataset,
    variable_names: Optional[list[str]] = None,
    level: Optional[list[int]] = None,
    normalization_param_dict: Optional[Dict] = None,
    mean_tensor: Optional[torch.Tensor] = None,
    std_tensor: Optional[torch.Tensor] = None,
) -> xr.Dataset:
    """ "
    input_tensor: (C, T, H, W), the return xarr will have the same time as the xarr_meta.
    """

    if variable_names is None:
        variable_names = list(xarr_meta.data_vars.keys())
    if level is None:
        level = list(xarr_meta["level"].values)

    selected_xarr = (
        xarr_meta[variable_names]
        .transpose("time", "level", "latitude", "longitude")
        .sel(level=level)
    )

    if normalization_param_dict is not None:
        mean_tensor, std_tensor = precompute_mean_std(
            normalization_param_dict, variable_names
        )
    if mean_tensor is not None:
        input_tensor = inverse_normalize_transform_3D(
            input_tensor,
            mean_tensor.to(input_tensor.device),
            std_tensor.to(input_tensor.device),
        )

    if input_tensor.is_cuda:
        input_tensor = input_tensor.cpu()

    cur_idx = 0
    return_xarr = xr.Dataset()
    for var_name in variable_names:
        if "level" in selected_xarr[var_name].dims:
            # dims = ('time', 'level', 'latitude', 'longitude')
            # coords = {'time': selected_xarr['time'], 'level': selected_xarr['level'], 'latitude': selected_xarr['latitude'], 'longitude': selected_xarr['longitude']}
            dims = selected_xarr[var_name].dims
            coords = selected_xarr[var_name].coords
            data_array = xr.DataArray(
                input_tensor[cur_idx : cur_idx + len(level)].permute(1, 0, 2, 3),
                dims=dims,
                coords=coords,
            )
            cur_idx += len(level)
        else:
            # dims = ('time', 'latitude', 'longitude')
            # coords = {'time': selected_xarr['time'], 'latitude': selected_xarr['latitude'], 'longitude': selected_xarr['longitude']}
            dims = selected_xarr[var_name].dims
            coords = selected_xarr[var_name].coords
            data_array = xr.DataArray(input_tensor[cur_idx], dims=dims, coords=coords)
            cur_idx += 1
        return_xarr[var_name] = data_array
    assert cur_idx == input_tensor.shape[0], "Mismatch in the number of variables."

    return return_xarr


def filter_time_range(time_index, num_samples_per_month, enforce_year=None):
    """
    Selects sample timestamps from a given DatetimeIndex, excluding the last day of each month.

    For each month in the time_index, the function computes num_samples_per_month
    days evenly spaced between the 1st day and just before the last day of the month.
    For each of these selected days, two timestamps are created: one at midnight (00:00)
    and one at noon (12:00).

    When enforce_year is provided, the time_index is first filtered to only include
    timestamps already in that year, and generated timestamps that fall outside of the
    original time range are skipped.

    Parameters:
      time_index (pd.DatetimeIndex): A Pandas DatetimeIndex covering the period of interest.
      num_samples_per_month (int): Number of days to sample per month.
      enforce_year (str or int, optional): If provided, only timestamps with this year are used,
                                             and the returned timestamps will use this year.

    Returns:
      pd.DatetimeIndex: A DatetimeIndex containing the selected timestamps.
    """
    selected_times = []
    # Determine the allowed end date from the input time_index.
    max_date = time_index.max()

    if enforce_year is not None:
        enforced_year = int(enforce_year)
        # Filter the time_index to only include timestamps that are already in the enforced year.
        filtered_index = time_index[time_index.year == enforced_year]

        # Group by month only (since all timestamps now have the enforced year)
        groups = {}
        for ts in filtered_index:
            groups.setdefault(ts.month, []).append(ts)

        for mo in sorted(groups.keys()):
            # Get the last day of the month for the enforced year
            _, last_day = calendar.monthrange(enforced_year, mo)
            # Compute sample days evenly spaced from 1 to last_day (excluding the last day)
            sample_days = np.linspace(
                1, last_day, num_samples_per_month, endpoint=False
            )
            sample_days = np.round(sample_days).astype(int)
            sample_days[0] = 1  # ensure the first day is always selected
            for day in sample_days:
                for hour in [0, 12]:
                    try:
                        dt = pd.Timestamp(
                            year=enforced_year, month=mo, day=day, hour=hour
                        )
                        # Only add the timestamp if it doesn't exceed the maximum date
                        if dt <= max_date:
                            selected_times.append(dt)
                    except Exception as e:
                        print(
                            f"Skipping invalid date: {enforced_year}-{mo}-{day} at {hour}:00 due to error: {e}"
                        )
    else:
        # Group by both year and month using the original time_index
        groups = {}
        for ts in time_index:
            key = (ts.year, ts.month)
            groups.setdefault(key, []).append(ts)

        for yr, mo in sorted(groups.keys()):
            _, last_day = calendar.monthrange(yr, mo)
            sample_days = np.linspace(
                1, last_day, num_samples_per_month, endpoint=False
            )
            sample_days = np.round(sample_days).astype(int)
            sample_days[0] = 1
            for day in sample_days:
                for hour in [0, 12]:
                    try:
                        dt = pd.Timestamp(year=yr, month=mo, day=day, hour=hour)
                        if dt <= max_date:
                            selected_times.append(dt)
                    except Exception as e:
                        print(
                            f"Skipping invalid date: {yr}-{mo}-{day} at {hour}:00 due to error: {e}"
                        )

    return pd.DatetimeIndex(sorted(selected_times))
