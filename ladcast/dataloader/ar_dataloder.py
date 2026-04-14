from typing import List, Optional

import numpy as np
import torch
import xarray as xr
import zarr
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


# Canonical dim aliases used by LaDCast:  C / time / H / W
_DIM_ALIASES = {
    "C": ("C", "channel", "level", "var"),
    "time": ("time",),
    "H": ("H", "height", "lat", "latitude"),
    "W": ("W", "width", "lon", "longitude"),
}


def _resolve_canonical(name: str) -> Optional[str]:
    """Return canonical dim name (C / time / H / W) for an input name, or None."""
    for canon, aliases in _DIM_ALIASES.items():
        if name in aliases:
            return canon
    return None


def _open_zarr_array(zarr_path: str, var_name: str):
    """Open the underlying zarr array for a variable, bypassing xarray.

    Returns
    -------
    zarr_arr : zarr.Array
        The raw zarr array (lazy, just metadata).
    zarr_dims : list[str]
        The original dim names from `_ARRAY_DIMENSIONS` attr (xarray-zarr
        convention), e.g. ['channel', 'time', 'lat', 'lon'].
    canonical_axis_map : dict[str, int]
        Mapping from canonical names ('C', 'time', 'H', 'W') to axis indices.
    """
    z = zarr.open(zarr_path, mode="r")
    # Resolve variable name (handle latent vs latents alias)
    candidates = [var_name, var_name.lower(), var_name.lower().rstrip("s")]
    if var_name in ("latent", "latents"):
        candidates.extend(["latent", "latents"])
    arr = None
    for name in candidates:
        if name in z:
            arr = z[name]
            break
    if arr is None:
        raise KeyError(
            f"Variable '{var_name}' not found in zarr store at {zarr_path}. "
            f"Candidates tried: {candidates}"
        )

    zarr_dims = list(arr.attrs.get("_ARRAY_DIMENSIONS", []))
    if not zarr_dims:
        raise ValueError(
            f"Zarr variable '{var_name}' has no '_ARRAY_DIMENSIONS' attribute. "
            "Cannot determine dim ordering for direct zarr access."
        )

    canonical_axis_map = {}
    for i, d in enumerate(zarr_dims):
        canon = _resolve_canonical(d)
        if canon is not None:
            canonical_axis_map[canon] = i

    missing = [d for d in ("C", "time", "H", "W") if d not in canonical_axis_map]
    if missing:
        # Try to expand C if the array is 3-D (time, H, W) — we'll add a virtual axis later
        if missing == ["C"] and len(zarr_dims) == 3:
            pass  # handled in dataset
        else:
            raise ValueError(
                f"Could not map all canonical dims. zarr_dims={zarr_dims}, "
                f"resolved={canonical_axis_map}, missing={missing}"
            )

    return arr, zarr_dims, canonical_axis_map


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
    # -- 1) Open xarray ONLY to extract metadata (time coords, time slicing) --
    #    We then close it immediately so its lazy/zarr cache layers can't grow.
    xarr = xr.open_dataset(ds_path, engine=xr_engine, chunks=None)
    xarr = _normalize_zarr_dataset(xarr)
    xarr = xarr.sel(time=slice(start_date, end_date))

    # Resolve variable name
    var_name = var_name.strip()
    if var_name not in xarr:
        available_vars = list(xarr.data_vars)
        alias_priority = [var_name, var_name.lower(), var_name.lower().rstrip("s")]
        explicit_aliases = {"latents": "latent", "latent": "latents"}
        alias_priority.extend(explicit_aliases.get(name, "") for name in alias_priority)
        alias_priority = [name for name in alias_priority if name]
        matched = next((name for name in alias_priority if name in available_vars), None)
        if matched is None:
            normalized_target = var_name.lower().rstrip("s")
            fallback_candidates = [
                name for name in available_vars
                if name.lower().rstrip("s") == normalized_target
            ]
            if len(fallback_candidates) == 1:
                matched = fallback_candidates[0]
        if matched is None:
            raise KeyError(
                f"Variable '{var_name}' not found in dataset. "
                f"Available vars: {available_vars}."
            )
        print(
            f"[prepare_ar_dataloader] var_name '{var_name}' not found, "
            f"auto-using alias '{matched}'."
        )
        var_name = matched

    data = xarr[var_name]

    # Compute the integer time-slice that `xarr.sel(time=slice(start, end))`
    # selected, expressed as offsets into the ORIGINAL zarr time axis.
    # We pull the *full* time coord straight from zarr (small 1-D array)
    # to avoid having to re-open xarray.
    selected_time = data.time.values.copy()

    # Read the full time coord directly from zarr — small (~1-D)
    _z_root = zarr.open(ds_path, mode="r")
    full_time = np.asarray(_z_root["time"][:])

    first_match = int(np.searchsorted(full_time, selected_time[0]))
    last_match = int(np.searchsorted(full_time, selected_time[-1])) + 1
    if not (
        0 <= first_match < len(full_time)
        and full_time[first_match] == selected_time[0]
    ):
        raise ValueError(
            f"start time {selected_time[0]} not found exactly in zarr time coord"
        )
    time_start_in_zarr = first_match
    time_stop_in_zarr = last_match

    # Save the time values we need (post date-slicing) — small, fits in memory
    selected_time_values = selected_time

    # Close & release xarray references — we only need raw zarr from here on
    try:
        xarr.close()
    except Exception:
        pass
    del xarr, data, full_time, selected_time, _z_root

    # -- 2) Open the underlying zarr array directly (no xarray cache) --
    zarr_arr, zarr_dims, canonical_axis_map = _open_zarr_array(ds_path, var_name)

    if not profiling:
        tmp_dataset = XarrayDataset3D(
            zarr_arr=zarr_arr,
            zarr_dims=zarr_dims,
            canonical_axis_map=canonical_axis_map,
            time_values=selected_time_values,
            time_start_in_zarr=time_start_in_zarr,
            time_stop_in_zarr=time_stop_in_zarr,
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
    """Direct-zarr backed dataset (avoids xarray's leaky lazy-loading cache).

    Each ``__getitem__`` reads a small slice straight from the zarr array
    (~ a couple of (C, H, W) frames) and returns it as a plain numpy array.
    Zarr does NOT cache slice objects the way xarray does, so memory usage
    stays bounded at roughly batch_size × frame_size.

    return shape: (C, T, H, W)
    """

    def __init__(
        self,
        zarr_arr,                                # zarr.Array (lazy reference)
        zarr_dims: List[str],                    # original dim names from zarr
        canonical_axis_map: dict,                # {'C': i, 'time': j, 'H': k, 'W': l}
        time_values: np.ndarray,                 # 1-D datetime array (post-date-slice)
        time_start_in_zarr: int,                 # index of time_values[0] in raw zarr
        time_stop_in_zarr: int,                  # one past index of time_values[-1]
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
        self._zarr = zarr_arr
        self._zarr_dims = zarr_dims
        self._axis_map = canonical_axis_map
        self._zarr_ndim = len(zarr_dims)
        self._has_C_axis = "C" in canonical_axis_map

        # Cache axis indices for hot-path use
        self._time_axis = canonical_axis_map["time"]
        # Permutation so that after read+transpose the result is (C, time, H, W)
        if self._has_C_axis:
            self._target_axes_in_zarr = (
                canonical_axis_map["C"],
                canonical_axis_map["time"],
                canonical_axis_map["H"],
                canonical_axis_map["W"],
            )
        else:
            # No C axis — we'll insert one at axis 0 after read
            self._target_axes_in_zarr = (
                canonical_axis_map["time"],
                canonical_axis_map["H"],
                canonical_axis_map["W"],
            )

        self._time_values = time_values.copy()
        # Apply truncate_first + sampling_interval to time values too
        self._time_values = self._time_values[truncate_first::sampling_interval]
        # Map subsampled idx -> raw zarr time idx:
        #     zarr_t = time_start_in_zarr + truncate_first + idx * sampling_interval
        self._time_offset = time_start_in_zarr + truncate_first
        self._sampling_interval = sampling_interval

        # Optionally pre-load entire data into memory (only if user asks AND it fits)
        self._np_data = None
        if load_in_memory:
            print(
                "[XarrayDataset3D] load_in_memory=True — reading full zarr into RAM. "
                "This will fail (OOM) if the dataset exceeds available memory."
            )
            full_slice = self._zarr_slice_for_time_range(
                time_start_in_zarr + truncate_first,
                time_stop_in_zarr,
                sampling_interval,
            )
            self._np_data = self._read_zarr_to_canonical(full_slice)
            print(
                f"[XarrayDataset3D] in-memory shape={self._np_data.shape}, "
                f"dtype={self._np_data.dtype}, "
                f"size={self._np_data.nbytes / 1024**2:.1f} MB"
            )
            # Override accessor: reads come from numpy
            self._time_offset = 0  # already pre-truncated
            self._sampling_interval = 1

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
            self.length = (
                len(self._time_values) - self.full_seq_len - truncate_first + 1
            )
            print(
                f"[XarrayDataset3D] using direct zarr access. "
                f"Subsampled time len={len(self._time_values)}, "
                f"calc length={self.length}, full_seq_len={self.full_seq_len}"
            )
        else:
            self.length = length - self.full_seq_len - truncate_first + 1

    # ---- internal helpers ----------------------------------------------------

    def _zarr_slice_for_time_range(self, t_start, t_stop, t_step):
        """Build an indexing tuple for the zarr array that selects all C/H/W
        and time[t_start:t_stop:t_step]."""
        slicer = [slice(None)] * self._zarr_ndim
        slicer[self._time_axis] = slice(t_start, t_stop, t_step)
        return tuple(slicer)

    def _read_zarr_to_canonical(self, slicer) -> np.ndarray:
        """Read from zarr and transpose to canonical (C, T, H, W) layout."""
        arr = self._zarr[slicer]  # numpy ndarray (uncached read)
        if self._has_C_axis:
            arr = arr.transpose(self._target_axes_in_zarr)
        else:
            # (time, H, W) → (1, time, H, W)
            arr = arr.transpose(self._target_axes_in_zarr)
            arr = arr[np.newaxis, ...]
        return arr

    def _read_window(self, sub_idx_start: int, sub_idx_stop: int, sub_step: int) -> np.ndarray:
        """Read [:, sub_idx_start:sub_idx_stop:sub_step, :, :] in subsampled space.

        Returns an array of shape (C, T_window, H, W).
        """
        if self._np_data is not None:
            # Pre-loaded numpy fast path
            return self._np_data[
                :, sub_idx_start:sub_idx_stop:sub_step, :, :
            ]
        # Translate subsampled indices → raw zarr time indices
        si = self._sampling_interval
        to = self._time_offset
        zarr_start = to + sub_idx_start * si
        zarr_stop = to + sub_idx_stop * si
        zarr_step = sub_step * si
        slicer = self._zarr_slice_for_time_range(zarr_start, zarr_stop, zarr_step)
        return self._read_zarr_to_canonical(slicer)

    # ---- Dataset interface ---------------------------------------------------

    @property
    def latent_H(self) -> int:
        """Return the H (latitude) dimension size of the underlying data."""
        return int(self._zarr.shape[self._axis_map["H"]])

    @property
    def latent_W(self) -> int:
        """Return the W (longitude) dimension size of the underlying data."""
        return int(self._zarr.shape[self._axis_map["W"]])

    def _preprocess_data(self, data):
        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        return self.transform(data)

    def __len__(self):
        return self.length

    def _get_return_timestamp(self, idx):
        input_end_idx = idx + (self.input_seq_len - 1) * self.interval_between_pred
        pred_start_idx = input_end_idx + self.interval_between_pred
        return (
            self._time_values[
                idx : (input_end_idx + 1) : self.interval_between_pred
            ],
            self._time_values[
                pred_start_idx : (
                    pred_start_idx + self.len_rest_after_first_pred_point + 1
                ) : self.interval_between_pred,
            ],
        )

    def __getitem__(self, idx):
        input_end_idx = (
            idx + (self.input_seq_len - 1) * self.interval_between_pred
        )
        pred_start_idx = input_end_idx + self.interval_between_pred
        pred_stop_idx = pred_start_idx + self.len_rest_after_first_pred_point + 1

        input_arr = self._read_window(
            idx, input_end_idx + 1, self.interval_between_pred
        )
        pred_arr = self._read_window(
            pred_start_idx, pred_stop_idx, self.interval_between_pred
        )
        return (
            self._preprocess_data(input_arr),
            self._preprocess_data(pred_arr),
            convert_datetime_to_int(self._time_values[idx]),
        )
