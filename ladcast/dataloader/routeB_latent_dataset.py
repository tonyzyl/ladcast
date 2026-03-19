import json
import os
from typing import Dict, Optional, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class RouteBLatentDataset(Dataset):
    """
    Minimal AR latent dataset for RouteB.

    Returns:
        {
          "x_in":  (input_seq_len, C, H, W),
          "x_out": (return_seq_len, C, H, W),
          "index": int,
          "time_in": list[str] (optional),
          "time_out": list[str] (optional),
        }
    """

    def __init__(
            self,
            latent_path: str,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            input_seq_len: int = 1,
            return_seq_len: int = 1,
            interval_between_pred: int = 1,
            normalize: bool = False,
            latent_norm_json: Optional[str] = None,
            return_time: bool = True,
    ) -> None:
        super().__init__()

        self.latent_path = os.path.expanduser(latent_path)
        self.input_seq_len = int(input_seq_len)
        self.return_seq_len = int(return_seq_len)
        self.interval_between_pred = int(interval_between_pred)
        self.normalize = bool(normalize)
        self.return_time = bool(return_time)

        if self.input_seq_len <= 0:
            raise ValueError("input_seq_len must be > 0")
        if self.return_seq_len <= 0:
            raise ValueError("return_seq_len must be > 0")
        if self.interval_between_pred <= 0:
            raise ValueError("interval_between_pred must be > 0")

        if not os.path.exists(self.latent_path):
            raise FileNotFoundError(f"Latent zarr not found: {self.latent_path}")

        # Metadata scan in main process only
        ds = xr.open_zarr(self.latent_path, consolidated=False)
        if "latent" not in ds:
            raise KeyError("Expected variable 'latent' in zarr dataset")

        da = ds["latent"]
        if start_time is not None or end_time is not None:
            da = da.sel(time=slice(start_time, end_time))

        if da.sizes.get("time", 0) <= 0:
            raise ValueError("Selected latent dataset has zero time length.")

        if tuple(da.dims) != ("time", "channel", "lat", "lon"):
            raise RuntimeError(f"Unexpected latent dims: {da.dims}")

        self.times = da["time"].values
        self.num_time = int(da.sizes["time"])
        self.num_channels = int(da.sizes["channel"])
        self.h = int(da.sizes["lat"])
        self.w = int(da.sizes["lon"])

        horizon = (
                self.input_seq_len
                + self.interval_between_pred
                + self.return_seq_len
                - 1
        )
        self.max_start = self.num_time - horizon
        if self.max_start <= 0:
            raise ValueError(
                f"Not enough time steps ({self.num_time}) for input_seq_len={self.input_seq_len}, "
                f"interval_between_pred={self.interval_between_pred}, return_seq_len={self.return_seq_len}."
            )

        if self.normalize:
            if latent_norm_json is None:
                raise ValueError("normalize=True requires latent_norm_json")
            latent_norm_json = os.path.expanduser(latent_norm_json)
            if not os.path.exists(latent_norm_json):
                raise FileNotFoundError(f"latent_norm_json not found: {latent_norm_json}")
            with open(latent_norm_json, "r", encoding="utf-8") as f:
                norm = json.load(f)
            mean = np.asarray(norm["mean"], dtype=np.float32)
            std = np.asarray(norm["std"], dtype=np.float32)
            if mean.shape[0] != self.num_channels or std.shape[0] != self.num_channels:
                raise ValueError(
                    f"Norm channels mismatch: dataset={self.num_channels}, "
                    f"mean={mean.shape[0]}, std={std.shape[0]}"
                )
            std = np.clip(std, 1e-6, None)
            self.mean = mean[:, None, None]
            self.std = std[:, None, None]
        else:
            self.mean = None
            self.std = None

        # 保存时间切片参数供懒加载用
        self.start_time = start_time
        self.end_time = end_time
        # 关闭共享对象，不保存da
        self.latent_da = None
        try:
            ds.close()
        except Exception:
            pass

    def _get_da(self):
        if self.latent_da is None:
            ds = xr.open_zarr(self.latent_path, consolidated=False)
            da = ds["latent"]
            if self.start_time is not None or self.end_time is not None:
                da = da.sel(time=slice(self.start_time, self.end_time))
            self.latent_da = da
        return self.latent_da

    def __len__(self) -> int:
        return self.max_start

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # x: (S, C, H, W)
        return (x - self.mean[None, ...]) / self.std[None, ...]

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, list[str]]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        in_start = idx
        in_end = idx + self.input_seq_len

        out_start = in_end - 1 + self.interval_between_pred
        out_end = out_start + self.return_seq_len

        x_in = self._get_da().isel(time=slice(in_start, in_end)).values.astype(np.float32)
        x_out = self._get_da().isel(time=slice(out_start, out_end)).values.astype(np.float32)

        if self.normalize:
            x_in = self._normalize(x_in)
            x_out = self._normalize(x_out)

        if not np.isfinite(x_in).all() or not np.isfinite(x_out).all():
            raise RuntimeError(f"Non-finite latent value at idx={idx}")

        sample: Dict[str, Union[torch.Tensor, int, list[str]]] = {
            "x_in": torch.from_numpy(x_in).to(torch.float32),
            "x_out": torch.from_numpy(x_out).to(torch.float32),
            "index": int(idx),
        }

        if self.return_time:
            time_in = [str(t) for t in self.times[in_start:in_end]]
            time_out = [str(t) for t in self.times[out_start:out_end]]
            sample["time_in"] = time_in
            sample["time_out"] = time_out

        return sample