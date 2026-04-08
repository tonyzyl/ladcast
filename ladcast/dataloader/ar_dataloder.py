from typing import Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from ladcast.dataloader.utils import get_inv_transform_3D, get_transform_3D


def convert_datetime_to_int(dt: np.timedelta64) -> int:
    if isinstance(dt, np.datetime64):
        dt_hour = dt.astype("datetime64[h]")
        py_dt = dt_hour.tolist()
    else:
        py_dt = dt

    return int(py_dt.strftime("%Y%m%d%H"))


def _normalize_zarr_dataset(xarr: xr.Dataset) -> xr.Dataset:
    """Rename dimensions/variables to the canonical names expected by LaDCast:
    dims:  (C, time, H, W)
    var:   latents
    """
    # Dimension rename: channel→C, lat→H, lon→W
    _dim_map = {"channel": "C", "lat": "H", "lon": "W"}
    _rename = {k: v for k, v in _dim_map.items() if k in xarr.dims}
    # Variable rename: latent→latents
    if "latent" in xarr.data_vars and "latents" not in xarr.data_vars:
        _rename["latent"] = "latents"
    if _rename:
        xarr = xarr.rename(_rename)
    return xarr


def prepare_ar_dataloader(
    ds_path: str,  # path to zarr file
    start_date: str,
    end_date: str,
    xr_engine: Optional[str] = "zarr",
    var_name: str = "latent",
    transform: callable = None,
    transform_args: dict = None,
    input_seq_len: int = 1,
    return_seq_len: int = 1,
    truncate_first: int = 0,
    sampling_interval: int = 1,
    interval_between_pred: int = 1,
    data_augmentation: Optional[bool] = False,
    batch_size: Optional[int] = 1,
    shuffle: Optional[bool] = False,
    num_workers: Optional[int] = 0,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    dask_threads: Optional[int] = None,
    profiling: Optional[bool] = False,
    load_in_memory: Optional[bool] = False,
):
    xarr = xr.open_dataset(ds_path, engine=xr_engine, chunks=None)
    xarr = _normalize_zarr_dataset(xarr)
    xarr = xarr.sel(time=slice(start_date, end_date))
    var_name = var_name.strip()
    if var_name not in xarr:
        available_vars = list(xarr.data_vars)
        alias_priority = [var_name, var_name.lower(), var_name.lower().rstrip("s")]
        explicit_aliases = {
            "latents": "latent",
            "latent": "latents",
        }
        alias_priority.extend(explicit_aliases.get(name, "") for name in alias_priority)
        alias_priority = [name for name in alias_priority if name]
        matched = next((name for name in alias_priority if name in available_vars), None)
        if matched is not None:
            print(
                f"[prepare_ar_dataloader] var_name '{var_name}' not found, "
                f"auto-using alias '{matched}'."
            )
            var_name = matched
        else:
            normalized_target = var_name.lower().rstrip("s")
            fallback_candidates = [
                name
                for name in available_vars
                if name.lower().rstrip("s") == normalized_target
            ]
            if len(fallback_candidates) == 1:
                matched = fallback_candidates[0]
                print(
                    f"[prepare_ar_dataloader] var_name '{var_name}' not found, "
                    f"auto-using closest match '{matched}'."
                )
                var_name = matched
            else:
                hint = (
                    "This dataset does not contain the requested variable name exactly. "
                    "Check singular/plural naming (e.g. 'latent' vs 'latents'), "
                    "or set train_dataloader.var_name to one of the available vars."
                )
                raise KeyError(
                    f"Variable '{var_name}' not found in dataset. "
                    f"Available vars: {available_vars}. {hint}"
                )

    data = xarr[var_name]
    dim_alias = {
        "C": ("C", "channel", "level", "var"),
        "H": ("H", "height", "lat", "latitude"),
        "W": ("W", "width", "lon", "longitude"),
    }

    rename_map = {}
    for target_dim, candidates in dim_alias.items():
        if target_dim in data.dims:
            continue
        matched_dim = next((name for name in candidates if name in data.dims), None)
        if matched_dim is not None and matched_dim != target_dim:
            rename_map[matched_dim] = target_dim

    if rename_map:
        data = data.rename(rename_map)

    if "C" not in data.dims and all(dim in data.dims for dim in ("time", "H", "W")):
        data = data.expand_dims("C")

    missing_dims = [dim for dim in ("C", "time", "H", "W") if dim not in data.dims]
    if missing_dims:
        raise ValueError(
            f"Variable '{var_name}' missing required dimensions {missing_dims}. "
            f"Current dims: {data.dims}"
        )

    data = data.transpose("C", "time", "H", "W")
    # xarr = xarr.chunk(chunks={"time": 1})
    if not profiling:
        tmp_dataset = XarrayDataset3D(
            data=data,
            transform=transform,
            transform_args=transform_args,
            input_seq_len=input_seq_len,
            return_seq_len=return_seq_len,
            truncate_first=truncate_first,
            sampling_interval=sampling_interval,
            interval_between_pred=interval_between_pred,
            data_augmentation=data_augmentation,
            load_in_memory=load_in_memory,
        )
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    return DataLoader(
        tmp_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )


class XarrayDataset3D(Dataset):
    """
    Sampling interval apply first.
    return shape: (C, T, H, W)
    """

    def __init__(
        self,
        data: xr.DataArray,  # (C, T, H, W)
        length: int = None,
        transform: str = "normalize",
        transform_args: dict = None,
        input_seq_len: int = 1,
        return_seq_len: int = 1,
        truncate_first: int = 0,
        sampling_interval: int = 1,
        interval_between_pred: int = 1,
        data_augmentation: Optional[bool] = False,
        load_in_memory: Optional[bool] = False,
    ):
        self.data = data[:, truncate_first:]
        self.data = data.isel(time=slice(truncate_first, None, sampling_interval))
        if load_in_memory:
            print("Load in memory is set to True, loading data...")
            self.data = self.data.load()
            print("Data loaded in memory")
        self.transform = get_transform_3D(transform, transform_args)
        self.inv_transform = get_inv_transform_3D(transform, transform_args)
        self.transform_args = transform_args or {}
        self.input_seq_len = input_seq_len
        self.return_seq_len = return_seq_len
        self.interval_between_pred = interval_between_pred
        self.data_augmentation = data_augmentation
        self.len_rest_after_first_pred_point = (
            return_seq_len - 1
        ) * interval_between_pred

        self.full_seq_len = (
            input_seq_len + return_seq_len - 1
        ) * interval_between_pred + 1
        if length is None:
            self.length = self.data.shape[1] - self.full_seq_len - truncate_first + 1
            print(
                f"Length not provided, Calculated: {self.length}. Full seq len: {self.full_seq_len}"
            )
        else:
            self.length = length - self.full_seq_len - truncate_first + 1

    def _preprocess_data(self, data):
        data = torch.from_numpy(data.to_numpy()).float()
        return self.transform(data)

    def __len__(self):
        return self.length

    def _get_return_timestamp(self, idx):
        input_end_idx = idx + (self.input_seq_len - 1) * self.interval_between_pred
        pred_start_idx = input_end_idx + self.interval_between_pred
        return (
            self.data[
                :, idx : (input_end_idx + 1) : self.interval_between_pred
            ].time.values,
            self.data[
                :,
                pred_start_idx : (
                    pred_start_idx + self.len_rest_after_first_pred_point + 1
                ) : self.interval_between_pred,
            ].time.values,
        )

    def __getitem__(self, idx):
        input_end_idx = (
            idx + (self.input_seq_len - 1) * self.interval_between_pred
        )  # the idx is the start of the input sequence
        pred_start_idx = input_end_idx + self.interval_between_pred
        return (
            self._preprocess_data(
                self.data[:, idx : (input_end_idx + 1) : self.interval_between_pred]
            ),
            self._preprocess_data(
                self.data[
                    :,
                    pred_start_idx : (
                        pred_start_idx + self.len_rest_after_first_pred_point + 1
                    ) : self.interval_between_pred,
                ]
            ),
            convert_datetime_to_int(self.data.time[idx].values),
        )
