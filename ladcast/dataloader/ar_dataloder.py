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


def prepare_ar_dataloader(
    ds_path: str,  # path to zarr file
    start_date: str,
    end_date: str,
    xr_engine: Optional[str] = "zarr",
    var_name: str = "latents",
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
    xarr = xr.open_dataset(ds_path, engine=xr_engine, chunks="auto")
    xarr = xarr.sel(time=slice(start_date, end_date))
    xarr = xarr.transpose("C", "time", "H", "W")
    # xarr = xarr.chunk(chunks={"time": 1})
    if not profiling:
        tmp_dataset = XarrayDataset3D(
            data=xarr[var_name],
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
