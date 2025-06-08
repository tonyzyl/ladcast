from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr


def get_lat_weights_from_lat_tensor(lat: torch.Tensor) -> torch.Tensor:
    """
    Replication of WeatherBench2's get_lat_weights
    Compute latitude/area weights for a batched latitude tensor.

    Args:
        lat (torch.Tensor): Tensor of latitudes in degrees with shape (B, L).

    Returns:
        torch.Tensor: Normalized latitude weights of shape (B, L).
    """
    lat_rad = torch.deg2rad(lat)  # shape (B, L)
    midpoints = (lat_rad[:, :-1] + lat_rad[:, 1:]) / 2

    B = lat_rad.shape[0]
    lower_bounds = torch.full(
        (B, 1), -torch.pi / 2, dtype=lat_rad.dtype, device=lat_rad.device
    )
    upper_bounds = torch.full(
        (B, 1), torch.pi / 2, dtype=lat_rad.dtype, device=lat_rad.device
    )

    bounds = torch.cat([lower_bounds, midpoints, upper_bounds], dim=1)
    cell_area = torch.sin(bounds[:, 1:]) - torch.sin(bounds[:, :-1])  # shape (B, L)

    mean_area = cell_area.mean(dim=1, keepdim=True)
    weights = cell_area / mean_area

    return weights


def get_normalized_lat_weights_based_on_cos(
    lat: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """requires lat in degrees"""
    if isinstance(lat, torch.Tensor):
        weights = torch.cos(torch.deg2rad(lat))
    else:
        weights = np.cos(np.deg2rad(lat))
    return weights / weights.mean()


@torch.no_grad()
def pointwise_crps_skill(
    forecast: torch.Tensor, truth: torch.Tensor, ensemble_dim: int
) -> torch.Tensor:
    """
    Modifled from WeatherBench2.metrics._pointwise_crps_skill to support tensor inputs
    truth tensor shall be broadcastable to forecast tensor
    """
    return torch.abs(truth - forecast).mean(dim=ensemble_dim)


@torch.no_grad()
def pointwise_crps_spread(forecast: torch.Tensor, ensemble_dim: int) -> torch.Tensor:
    """
    Modified from WeatherBench2.metrics._pointwise_crps_spread to support tensor inputs
    Computes CRPS spread in a memory-efficient way using sorted forecasts.
    The spread is computed as:
      spread = 2/(M*(M-1)) * sum_{i=1}^M [ (2*i - M - 1) * sorted_forecast[i] ]
    where M is the ensemble size.
    """
    n_ensemble = forecast.shape[ensemble_dim]
    if n_ensemble < 2:
        # If there's no ensemble spread then return a tensor of zeros.
        return torch.zeros_like(forecast.select(ensemble_dim, 0))

    # Sort the forecast along the ensemble dimension.
    sorted_forecast, _ = torch.sort(forecast, dim=ensemble_dim)

    # Create a weight vector with shape (n_ensemble,)
    # Using ranks from 1 to n_ensemble.
    weights = (
        2
        * (
            torch.arange(
                1, n_ensemble + 1, device=forecast.device, dtype=forecast.dtype
            )
        )
        - n_ensemble
        - 1
    )

    # Reshape weights for broadcasting:
    shape = [1] * forecast.ndim
    shape[ensemble_dim] = -1
    weights = weights.view(*shape)

    # Compute the weighted sum over the ensemble dimension.
    # Note: This is equivalent to taking the mean of (weighted forecast) and scaling appropriately.
    weighted_sum = (sorted_forecast * weights).sum(dim=ensemble_dim)

    return 2 * weighted_sum / (n_ensemble * (n_ensemble - 1))


@torch.no_grad()
def get_crps(
    forecast: torch.Tensor, truth: torch.Tensor, ensemble_dim: int = 0
) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for a given forecast and truth tensor.
    The ensemble dim is averaged out

    Args:
        spatial_start_dim (int): The starting dimension index for spatial dimensions in the truth tensor.
    """
    crps_skill = pointwise_crps_skill(forecast, truth, ensemble_dim)
    crps_spread = pointwise_crps_spread(forecast, ensemble_dim)

    return crps_skill - 0.5 * crps_spread


@torch.no_grad()
def get_acc(
    forecast: torch.Tensor,
    truth: torch.Tensor,
    climate: torch.Tensor,
    lat_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the Anomaly correlation coefficient (ACC),
    forecast, truth, climate and lat_weight tensors should be broadcastable
    based on WB2 implementation
    """
    forecast_anom = forecast - climate
    truth_anom = truth - climate
    # spatial average
    if lat_weight is not None:
        return (forecast_anom * truth_anom * lat_weight).nanmean(dim=(-2, -1)) / (
            torch.sqrt(
                (forecast_anom**2 * lat_weight).nanmean(dim=(-2, -1))
                * (truth_anom**2 * lat_weight).nanmean(dim=(-2, -1))
            )
        )
    else:
        return (forecast_anom * truth_anom).nanmean(dim=(-2, -1)) / (
            torch.sqrt(
                (forecast_anom**2).nanmean(dim=(-2, -1))
                * (truth_anom**2).nanmean(dim=(-2, -1))
            )
        )


def climatology_to_timeseries(
    ds, start_time, lead_time, interval=6, exclude_start=True
):
    """
    Parameters
    ----------
    ds : xarray.Dataset or DataArray
    start_time : str or pd.Timestamp
    lead_time : int
    interval : int, optional
    exclude_start : bool, optional
        If True (default), the `start_time` point will be omitted from the
        returned series.  If False, `start_time` is included.

    Returns
    -------
    xarray.Dataset or DataArray
    """
    if not {"dayofyear", "hour"}.issubset(ds.dims):
        raise ValueError("Dataset must have both 'dayofyear' and 'hour' dims")

    start = pd.to_datetime(start_time)
    end = start + pd.Timedelta(hours=int(lead_time))
    time_index = pd.date_range(start=start, end=end, freq=f"{int(interval)}h")

    if exclude_start:
        time_index = time_index[1:]

    nt = len(time_index)
    t = np.arange(nt)

    doy = xr.DataArray(
        time_index.dayofyear, coords={"t": t}, dims="t", name="dayofyear"
    )
    hr = xr.DataArray(time_index.hour, coords={"t": t}, dims="t", name="hour")

    ts = ds.sel(dayofyear=doy, hour=hr)

    ts = (
        ts.rename({"t": "time"})
        .assign_coords(time=("time", time_index))
        .transpose("time", ...)
    )

    if isinstance(ts, xr.Dataset):
        ts = ts.drop_vars(["dayofyear", "hour"], errors="ignore")
    else:  # DataArray
        ts = ts.reset_coords(["dayofyear", "hour"], drop=True)

    return ts
