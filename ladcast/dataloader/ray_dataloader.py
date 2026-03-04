import xarray as xr
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict, Any
import warnings

from ladcast.models.embeddings import convert_timestamp_to_int
from ladcast.dataloader.weather_dataset import weather_dataset_preprocess_batch
from ladcast.dataloader.utils import extract_raw_data_from_zarr

class ZarrLazyMapper:
    def __init__(
        self, 
        zarr_path: str, 
        surface_vars: List[str], 
        atmospheric_vars: List[str],
        preprocess: bool = False,
        mean: Optional[torch.Tensor] = None, # on cpu
        std: Optional[torch.Tensor] = None, # on cpu
        sst_channel_idx: Optional[int] = None,
        crop_south_pole: bool = True,
        incl_sur_pressure: bool = True,
    ):
        self.zarr_path = zarr_path
        self.surface_vars = surface_vars
        self.atmospheric_vars = atmospheric_vars
        self.ds = None 
        
        # preprocessing configs
        self.preprocess = preprocess
        self.mean = mean
        self.std = std
        self.sst_channel_idx = sst_channel_idx
        self.crop_south_pole = crop_south_pole
        self.incl_sur_pressure = incl_sur_pressure

        if self.preprocess:
            if self.mean is None or self.std is None:
                raise ValueError("Preprocessing enabled but mean/std not provided.")

        assert self.sst_channel_idx is not None, "For now we enforce sst_channel_idx."

    def _init_ds(self):
        if self.ds is None:
            # Open dataset lazily. 
            self.ds = xr.open_dataset(self.zarr_path, engine="zarr", chunks={})

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self._init_ds()
        
        timestamps = batch["timestamp"]
        
        try:
            ds_slice = self.ds.sel(time=timestamps)

            data_values = extract_raw_data_from_zarr(
                ds_slice,
                self.surface_vars,
                self.atmospheric_vars
                ) # (B, C, H, W)
            
            nan_mask = np.array([])
            if self.preprocess:
                tensor_batch = torch.from_numpy(data_values)
                tensor_batch, tensor_mask = weather_dataset_preprocess_batch(
                    tensor_batch,
                    self.mean,
                    self.std,
                    crop_south_pole=self.crop_south_pole,
                    sst_channel_idx=self.sst_channel_idx,
                    incl_sur_pressure=self.incl_sur_pressure
                )
                
                data_values = tensor_batch.numpy()
                nan_mask = tensor_mask.numpy()

        except KeyError as e:
            warnings.warn(f"One or more timestamps not found in dataset. Error: {e}. Skipping all.")
            return {"data": np.array([]), "timestamp": np.array([]), "nan_mask": np.array([])}
        except Exception as e:
            warnings.warn(f"Error processing batch {timestamps}: {e}")
            return {"data": np.array([]), "timestamp": np.array([]), "nan_mask": np.array([])}

        return {
            "data": data_values,
            "timestamp": np.array([convert_timestamp_to_int(pd.Timestamp(ts).strftime("%Y%m%d%H")) for ts in timestamps]),
            "nan_mask": nan_mask,
        }


def get_zarr_timestamps(zarr_path: str, start, end) -> List[Dict[str, Any]]:
    """
    Scans the Zarr dataset and returns a list of dictionaries containing timestamps within the start and end range.
    Each dictionary is {"timestamp": np.datetime64}.
    start and end can be int (year), str (parseable date), pd.Timestamp, np.datetime64, or np.timedelta64 (for end, relative to start).
    Raises ValueError if invalid format.
    
    Examples:
    - start=1979, end=1980  # years as integers
    - start="1979-01-01", end="1980-12-31"  # ISO date strings
    - start="19790101T00", end="19801231T23"  # custom string formats
    - start=pd.Timestamp("1979-01-01"), end=pd.Timestamp("1980-12-31")  # pd.Timestamp objects
    - start=1979, end=np.timedelta64(1, 'Y')  # end as timedelta from start
    """
    def parse_date(value):
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(value)
        elif isinstance(value, str):
            try:
                return pd.to_datetime(value)
            except Exception as e:
                raise ValueError(f"Invalid string date format: {value}, error: {e}")
        elif isinstance(value, int):
            return pd.Timestamp(year=value, month=1, day=1)
        else:
            raise ValueError(f"Unsupported date format: {type(value)} for {value}")
    
    start_date = parse_date(start)
    
    if isinstance(end, np.timedelta64):
        end_date = start_date + pd.Timedelta(end)
    else:
        end_date = parse_date(end)
    
    ds = xr.open_dataset(zarr_path, engine="zarr", chunks={})
    
    # Load time values (usually small enough to fit in memory)
    time_values = ds.time.values
    
    selected_times = []
    for t in time_values:
        ts = pd.Timestamp(t)
        if start_date <= ts <= end_date:
            selected_times.append({"timestamp": t})
                
    return selected_times
