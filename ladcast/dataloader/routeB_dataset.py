import os
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


PRESSURE_VARS: List[str] = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

SURFACE_VARS: List[str] = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]

# 固定通道顺序：
# 1. geopotential 13 levels
# 2. specific_humidity 13 levels
# 3. temperature 13 levels
# 4. u_component_of_wind 13 levels
# 5. v_component_of_wind 13 levels
# 6. 10m_u_component_of_wind
# 7. 10m_v_component_of_wind
# 8. 2m_temperature
# 9. mean_sea_level_pressure
# 10. total_precipitation_6hr
#
# 最终 shape: (70, 121, 240)


class ERA5RouteBDataset(Dataset):
    def __init__(
            self,
            ds_path: str = "~/data/ERA5_ladcast_routeB_1979_2024.zarr",
            norm_path: str = "~/ladcast/static/ERA5_routeB_normal_1979_2017.json",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            normalize: bool = True,
            return_time: bool = True,
    ) -> None:
        super().__init__()

        self.ds_path = os.path.expanduser(ds_path)
        self.norm_path = os.path.expanduser(norm_path)
        self.normalize = normalize
        self.return_time = return_time

        if not os.path.exists(self.ds_path):
            raise FileNotFoundError(f"Dataset not found: {self.ds_path}")

        self.ds = xr.open_zarr(self.ds_path, consolidated=True)

        if start_time is not None or end_time is not None:
            self.ds = self.ds.sel(time=slice(start_time, end_time))

        if self.ds.sizes["time"] <= 0:
            raise ValueError("Selected dataset has zero time length.")

        # 基本检查
        for var in PRESSURE_VARS + SURFACE_VARS:
            if var not in self.ds:
                raise KeyError(f"Missing variable in dataset: {var}")

        self.levels = self.ds["level"].values
        self.times = self.ds["time"].values

        if normalize:
            if not os.path.exists(self.norm_path):
                raise FileNotFoundError(f"Norm json not found: {self.norm_path}")
            with open(self.norm_path, "r", encoding="utf-8") as f:
                self.norm_stats = json.load(f)
            self._check_norm_stats()
            self._build_norm_arrays()
        else:
            self.norm_stats = None
            self.pressure_mean = None
            self.pressure_std = None
            self.surface_mean = None
            self.surface_std = None

    def _check_norm_stats(self) -> None:
        for var in PRESSURE_VARS:
            if var not in self.norm_stats:
                raise KeyError(f"Missing pressure norm stats: {var}")
            mean = self.norm_stats[var]["mean"]
            std = self.norm_stats[var]["std"]
            if not isinstance(mean, list) or not isinstance(std, list):
                raise TypeError(f"Pressure norm stats for {var} must be list.")
            if len(mean) != len(self.levels) or len(std) != len(self.levels):
                raise ValueError(
                    f"Pressure norm stats for {var} length mismatch: "
                    f"expected {len(self.levels)}, got {len(mean)} / {len(std)}"
                )

        for var in SURFACE_VARS:
            if var not in self.norm_stats:
                raise KeyError(f"Missing surface norm stats: {var}")
            mean = self.norm_stats[var]["mean"]
            std = self.norm_stats[var]["std"]
            if isinstance(mean, list) or isinstance(std, list):
                raise TypeError(f"Surface norm stats for {var} must be scalar.")

    def _build_norm_arrays(self) -> None:
        # pressure 按变量顺序堆起来，shape = (5, 13)
        p_mean = []
        p_std = []
        for var in PRESSURE_VARS:
            p_mean.append(np.asarray(self.norm_stats[var]["mean"], dtype=np.float32))
            p_std.append(np.asarray(self.norm_stats[var]["std"], dtype=np.float32))
        self.pressure_mean = np.stack(p_mean, axis=0)  # (5, 13)
        self.pressure_std = np.stack(p_std, axis=0)    # (5, 13)

        # surface shape = (5,)
        s_mean = []
        s_std = []
        for var in SURFACE_VARS:
            s_mean.append(np.float32(self.norm_stats[var]["mean"]))
            s_std.append(np.float32(self.norm_stats[var]["std"]))
        self.surface_mean = np.asarray(s_mean, dtype=np.float32)  # (5,)
        self.surface_std = np.asarray(s_std, dtype=np.float32)    # (5,)

    def __len__(self) -> int:
        return int(self.ds.sizes["time"])

    def _get_pressure_stack(self, idx: int) -> np.ndarray:
        # 输出 shape: (5, 13, 121, 240)
        arrs = []
        for var in PRESSURE_VARS:
            x = self.ds[var].isel(time=idx).values.astype(np.float32)  # (13, 121, 240)
            arrs.append(x)
        out = np.stack(arrs, axis=0)  # (5, 13, 121, 240)
        return out

    def _get_surface_stack(self, idx: int) -> np.ndarray:
        # 输出 shape: (5, 121, 240)
        arrs = []
        for var in SURFACE_VARS:
            x = self.ds[var].isel(time=idx).values.astype(np.float32)  # (121, 240)
            arrs.append(x)
        out = np.stack(arrs, axis=0)  # (5, 121, 240)
        return out

    def _normalize_pressure(self, x: np.ndarray) -> np.ndarray:
        # x: (5, 13, H, W)
        mean = self.pressure_mean[:, :, None, None]
        std = self.pressure_std[:, :, None, None]
        return (x - mean) / std

    def _normalize_surface(self, x: np.ndarray) -> np.ndarray:
        # x: (5, H, W)
        mean = self.surface_mean[:, None, None]
        std = self.surface_std[:, None, None]
        return (x - mean) / std

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        p = self._get_pressure_stack(idx)  # (5, 13, H, W)
        s = self._get_surface_stack(idx)   # (5, H, W)

        if self.normalize:
            p = self._normalize_pressure(p)
            s = self._normalize_surface(s)

        # pressure 展平成 (65, H, W)
        p = p.reshape(-1, p.shape[-2], p.shape[-1])  # (65, 121, 240)

        # surface 已经是 (5, H, W)
        x = np.concatenate([p, s], axis=0).astype(np.float32)  # (70, 121, 240)

        sample: Dict[str, Union[torch.Tensor, str, int]] = {
            "x": torch.from_numpy(x).to(torch.float32),
            "index": int(idx),
        }

        if self.return_time:
            sample["time"] = str(self.times[idx])

        return sample