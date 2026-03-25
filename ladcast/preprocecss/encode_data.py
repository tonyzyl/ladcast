import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr
from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from ladcast.dataloader.utils import precompute_mean_std, xarr_to_tensor
from ladcast.dataloader.weather_dataset import weather_dataset_preprocess_batch
from ladcast.models.DCAE import AutoencoderDC


# 可按需要补充/修改
VAR_NAME_ALIASES = {
    # pressure 常见短名 -> config 里的长名
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",

    # surface 常见短名 -> config 里的长名
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "sst": "sea_surface_temperature",
    "tp": "total_precipitation_6hr",
}


def normalize_var_name(name: str) -> str:
    """把 zarr 里的 var 名统一映射到 config 使用的变量名。"""
    name = str(name)
    return VAR_NAME_ALIASES.get(name, name)

def _decode_str_array(arr):
    out = []
    for x in np.array(arr):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out

def _find_time_slice_indices(time_array, start_date=None, end_date=None, time_origin=None):
    """
    在 zarr 的 time 数组中找到 [start_date, end_date] 对应的切片索引。

    支持:
    1) datetime64
    2) bytes / str
    3) 数值型 + attrs['units'] 形如 'hours since 1900-01-01 00:00:00'
    4) 纯整数时间轴，此时必须提供 time_origin，例如 1979-01-01T00

    返回:
        start_idx, end_idx_exclusive, sliced_time_array(datetime64[ns])
    """
    raw = np.array(time_array[:])
    attrs = {}
    try:
        attrs = dict(time_array.attrs)
    except Exception:
        pass

    # 情况 1：已经是 datetime64
    if np.issubdtype(raw.dtype, np.datetime64):
        t = raw.astype("datetime64[ns]")

    # 情况 2：字符串 / bytes
    elif raw.dtype.kind in ("U", "S", "O"):
        decoded = []
        for x in raw:
            if isinstance(x, bytes):
                decoded.append(x.decode("utf-8"))
            else:
                decoded.append(str(x))
        t = np.array(decoded, dtype="datetime64[ns]")

    # 情况 3：数值型
    elif np.issubdtype(raw.dtype, np.number):
        units = attrs.get("units", None)

        if isinstance(units, bytes):
            units = units.decode("utf-8")

        # 3a. CF 风格时间: "hours since 1900-01-01 00:00:00"
        if isinstance(units, str) and " since " in units:
            unit_part, base_part = units.split(" since ", 1)
            unit_part = unit_part.strip().lower()
            base = np.datetime64(base_part.strip())

            if unit_part in ["hour", "hours", "hr", "hrs", "h"]:
                delta = raw.astype("timedelta64[h]")
            elif unit_part in ["day", "days", "d"]:
                delta = raw.astype("timedelta64[D]")
            elif unit_part in ["minute", "minutes", "min", "mins", "m"]:
                delta = raw.astype("timedelta64[m]")
            elif unit_part in ["second", "seconds", "sec", "secs", "s"]:
                delta = raw.astype("timedelta64[s]")
            else:
                raise ValueError(f"Unsupported time unit in attrs['units']: {units}")

            t = (base + delta).astype("datetime64[ns]")

        # 3b. 没有 units，但给了 time_origin：按“小时偏移”解释
        elif time_origin is not None:
            base = np.datetime64(time_origin)
            t = (base + raw.astype("timedelta64[h]")).astype("datetime64[ns]")

        # 3c. 尝试按 Unix 时间戳推断
        else:
            v0 = float(raw[0])
            if v0 > 1e17:
                t = raw.astype("datetime64[ns]")
            elif v0 > 1e14:
                t = raw.astype("datetime64[us]").astype("datetime64[ns]")
            elif v0 > 1e11:
                t = raw.astype("datetime64[ms]").astype("datetime64[ns]")
            elif v0 > 1e8:
                t = raw.astype("datetime64[s]").astype("datetime64[ns]")
            else:
                raise ValueError(
                    "Numeric time array has no usable 'units' attribute and does not "
                    "look like a Unix timestamp. Please provide --time_origin."
                )
    else:
        raise ValueError(f"Unsupported time dtype: {raw.dtype}")

    if len(t) == 0:
        raise ValueError("Time array is empty")

    if start_date is None:
        start_idx = 0
    else:
        start_np = np.datetime64(start_date)
        start_idx = int(np.searchsorted(t, start_np, side="left"))

    if end_date is None:
        end_idx = len(t)
    else:
        end_np = np.datetime64(end_date)
        end_idx = int(np.searchsorted(t, end_np, side="right"))

    if start_idx >= end_idx:
        print("DEBUG time dtype:", t.dtype)
        print("DEBUG first 5 times:", t[:5])
        print("DEBUG last 5 times:", t[-5:])
        print("DEBUG requested:", start_date, end_date)
        raise ValueError(
            f"Empty time selection: start_date={start_date}, end_date={end_date}"
        )

    return start_idx, end_idx, t[start_idx:end_idx]


def _zarr_group_to_dataset(zarr_group_path: str, start_date=None, end_date=None, time_origin=None) -> xr.Dataset:
    """
    从原始 zarr group 中只读取指定时间范围的数据，并手工构造成 xarray.Dataset。

    支持:
    surface:
        data(time, var, latitude, longitude)
        latitude, longitude, time, var

    pressure:
        data(time, var, level, latitude, longitude)
        latitude, longitude, time, level, var
    """
    g = zarr.open_group(zarr_group_path, mode="r")

    required_common = ["data", "time", "latitude", "longitude", "var"]
    for k in required_common:
        if k not in g:
            raise ValueError(f"{zarr_group_path} missing required array: {k}")

    time_arr = g["time"]
    start_idx, end_idx, time = _find_time_slice_indices(
        time_arr, start_date, end_date, time_origin=time_origin
    )

    latitude = np.array(g["latitude"])
    longitude = np.array(g["longitude"])
    var_names = _decode_str_array(g["var"])

    data_zarr = g["data"]
    out = {}

    # surface: data(time, var, latitude, longitude)
    if "level" not in g:
        if len(data_zarr.shape) != 4:
            raise ValueError(
                f"{zarr_group_path} expected surface data with 4 dims "
                f"(time, var, latitude, longitude), got shape {data_zarr.shape}"
            )

        if data_zarr.shape[1] != len(var_names):
            raise ValueError(
                f"{zarr_group_path} var dimension mismatch: "
                f"data.shape[1]={data_zarr.shape[1]}, len(var)={len(var_names)}"
            )

        # 只切所需时间范围
        data = np.array(data_zarr[start_idx:end_idx, :, :, :])

        for i, raw_v in enumerate(var_names):
            out_name = normalize_var_name(raw_v)
            out[out_name] = xr.DataArray(
                data[:, i, :, :],
                dims=("time", "latitude", "longitude"),
                coords={
                    "time": time,
                    "latitude": latitude,
                    "longitude": longitude,
                },
            )

    # pressure: data(time, var, level, latitude, longitude)
    else:
        level = np.array(g["level"])

        if len(data_zarr.shape) != 5:
            raise ValueError(
                f"{zarr_group_path} expected pressure data with 5 dims "
                f"(time, var, level, latitude, longitude), got shape {data_zarr.shape}"
            )

        if data_zarr.shape[1] != len(var_names):
            raise ValueError(
                f"{zarr_group_path} var dimension mismatch: "
                f"data.shape[1]={data_zarr.shape[1]}, len(var)={len(var_names)}"
            )

        if data_zarr.shape[2] != len(level):
            raise ValueError(
                f"{zarr_group_path} level dimension mismatch: "
                f"data.shape[2]={data_zarr.shape[2]}, len(level)={len(level)}"
            )

        # 只切所需时间范围
        data = np.array(data_zarr[start_idx:end_idx, :, :, :, :])

        for i, raw_v in enumerate(var_names):
            out_name = normalize_var_name(raw_v)
            out[out_name] = xr.DataArray(
                data[:, i, :, :, :],
                dims=("time", "level", "latitude", "longitude"),
                coords={
                    "time": time,
                    "level": level,
                    "latitude": latitude,
                    "longitude": longitude,
                },
            )

    return xr.Dataset(out)


def open_era5_dataset(ds_path: str, start_date=None, end_date=None, time_origin=None) -> xr.Dataset:
    """
    支持:
    1) ds_path 是外层目录，里面有 pressure/ 和 surface/
    2) ds_path 直接指向单个原始 zarr group
    """
    path = Path(ds_path)
    pressure_path = path / "pressure"
    surface_path = path / "surface"

    if pressure_path.exists() and surface_path.exists():
        print(f"Detected split ERA5 store under: {ds_path}")
        print(f"  opening surface:  {surface_path}")
        print(f"  opening pressure: {pressure_path}")
        print(f"  selecting time range: {start_date} -> {end_date}")

        ds_surface = _zarr_group_to_dataset(
            str(surface_path), start_date, end_date, time_origin=time_origin
        )
        ds_pressure = _zarr_group_to_dataset(
            str(pressure_path), start_date, end_date, time_origin=time_origin
        )

        ds = xr.merge([ds_surface, ds_pressure], join="inner", compat="override")
        return ds

    if path.exists():
        print(f"Opening raw zarr group: {ds_path}")
        return _zarr_group_to_dataset(
            str(path), start_date, end_date, time_origin=time_origin
        )

    raise FileNotFoundError(f"Dataset path not found: {ds_path}")


@torch.no_grad()
def encode_latents_and_save_zarr_direct(
        ds,
        vae,
        zarr_path,
        channel_names,
        normalization_param_dict,
        static_conditioning_tensor,
):
    """
    Encodes latents from the provided xarray dataset directly using xarr_to_tensor
    and saves them to a Zarr store with the "time" coordinate.
    """
    accelerator = Accelerator()

    vae.eval()
    vae = accelerator.prepare(vae)
    device = accelerator.device
    print(f"device: {device}")

    static_conditioning_tensor = static_conditioning_tensor.to(device)

    all_times = ds.time.values
    total_samples = len(all_times)

    latents_tensor = torch.full(
        (total_samples, 84, 15, 30), float("nan"), dtype=torch.float32, device="cpu"
    )

    progress_bar = tqdm(range(total_samples), desc="Encoding Latents")

    for idx in progress_bar:
        single_time = all_times[idx]
        ds_single = ds.sel(time=[single_time])

        input_tensor = xarr_to_tensor(
            ds_single,
            variable_names=channel_names,
            add_static=False,
            normalization_param_dict=normalization_param_dict,
        ).permute(1, 0, 2, 3)  # shape: (1, C, H, W)

        input_tensor = input_tensor.to(device)

        latent = vae.encode(
            input_tensor,
            static_conditioning_tensor=static_conditioning_tensor.unsqueeze(0),
        ).latent

        latent_cpu = latent.squeeze(0).detach().cpu()
        latents_tensor[idx] = latent_cpu

    all_latents = latents_tensor.numpy()

    ds_latents = xr.Dataset(
        {"latents": (("time", "C", "H", "W"), all_latents)},
        coords={"time": all_times},
    )

    ds_latents.to_zarr(
        zarr_path,
        mode="w",
    )

    print(f"Latents and time successfully saved to {zarr_path}")
    accelerator.wait_for_everyone()


@torch.no_grad()
def encode_latents_and_save_zarr_hf_dataset(
        zarr_path,
        vae,
        channel_names,
        normalization_param_dict,
        static_conditioning_tensor,
        start_date,
        end_date,
        dataset_py_path="dataloader/weather_dataset.py",
):
    """
    Encodes latents using HuggingFace datasets iterable format and saves to Zarr.
    """
    accelerator = Accelerator(mixed_precision="no")

    vae.eval()
    vae = accelerator.prepare(vae)
    device = accelerator.device
    print(f"device: {device}")

    static_conditioning_tensor = static_conditioning_tensor.to(device)

    dataset = load_dataset(
        dataset_py_path,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, variable_names=channel_names
    )
    mean_tensor = mean_tensor[:, None, None].to(device)
    std_tensor = std_tensor[:, None, None].to(device)

    dataset = dataset.with_format("torch")

    start_dt = datetime.strptime(start_date, "%Y-%m-%dT%H")
    end_dt = datetime.strptime(end_date, "%Y-%m-%dT%H")
    delta = end_dt - start_dt
    total_hours = delta.days * 24 + delta.seconds // 3600 + 1

    print(f"Processing {total_hours} timesteps from {start_date} to {end_date}")

    all_times = [start_dt + timedelta(hours=i) for i in range(total_hours)]
    all_times_dt64 = np.array(all_times, dtype="datetime64[ns]")

    latents_tensor = torch.full(
        (total_hours, 84, 15, 30), float("nan"), dtype=torch.float32, device="cpu"
    )

    progress_bar = tqdm(range(total_hours), desc="Encoding Latents")

    for idx, sample in enumerate(dataset):
        if idx >= total_hours:
            break

        input_tensor = sample["data"].to(device)

        input_tensor, nan_mask = weather_dataset_preprocess_batch(
            input_tensor.unsqueeze(0),
            mean_tensor,
            std_tensor,
            crop_south_pole=True,
            sst_channel_idx=82,
            incl_sur_pressure=False,
        )

        B = input_tensor.shape[0]
        static_expanded = static_conditioning_tensor.expand(B, -1, -1, -1).clone()
        latent = vae.encode(
            input_tensor, static_conditioning_tensor=static_expanded
        ).latent
        latent_cpu = latent.squeeze(0).detach().cpu()

        latents_tensor[idx] = latent_cpu
        progress_bar.update(1)

    all_latents = latents_tensor.numpy()

    ds_latents = xr.Dataset(
        {"latents": (("time", "C", "H", "W"), all_latents)},
        coords={"time": all_times_dt64},
    )

    ds_latents.to_zarr(
        zarr_path,
        mode="w",
    )

    print(f"Latents and time successfully saved to {zarr_path}")
    accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Encode data to latent space and save to Zarr")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/encode_dataloader.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the encoded latents (overrides config)",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to input dataset (overrides config)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date in format 'YYYY-MM-DDThh' (overrides config)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in format 'YYYY-MM-DDThh' (overrides config)",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="tonyzyl/ladcast",
        help="HuggingFace repo name for model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="V0.1.X/DCAE",
        help="Model name/subfolder in repo",
    )
    parser.add_argument(
        "--normalization_json",
        type=str,
        default="static/ERA5_normal_1979_2017.json",
        help="Path to normalization parameters JSON",
    )
    parser.add_argument(
        "--lsm_path",
        type=str,
        default="static/240x121_land_sea_mask.pt",
        help="Path to land-sea mask tensor",
    )
    parser.add_argument(
        "--orography_path",
        type=str,
        default="static/240x121_orography.pt",
        help="Path to orography tensor",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["direct", "hf"],
        default="hf",
        help="Encoding method: 'direct' for xarray or 'hf' for HuggingFace datasets",
    )
    parser.add_argument(
        "--dataset_py_path",
        type=str,
        default="dataloader/weather_dataset.py",
        help="Path to dataset script for HF method",
    )
    parser.add_argument(
        "--time_origin",
        type=str,
        default="1979-01-01T00",
        help="Origin time for integer time axis, e.g. 1979-01-01T00",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    full_dataloader_config = config.pop("full_dataloader", OmegaConf.create())

    ds_path = args.ds_path or full_dataloader_config.get("ds_path")
    start_date = args.start_date or full_dataloader_config.get("start_date")
    end_date = args.end_date or full_dataloader_config.get("end_date")

    zarr_output_path = args.output or full_dataloader_config.get(
        "output_path", "zarr_output_path.zarr"
    )

    output_dir = Path(zarr_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    atmospheric_vars = full_dataloader_config.get("atmospheric_variables_name_list", [])
    surface_vars = full_dataloader_config.get("surface_variables_name_list", [])
    channel_names = list(atmospheric_vars) + list(surface_vars)

    print(f"Using channels: {channel_names}")

    if args.method == "direct":
        print(f"Loading dataset from {ds_path}")
        ds = open_era5_dataset(
            ds_path,
            start_date=start_date,
            end_date=end_date,
            time_origin=args.time_origin,
        )

        print("Loaded dataset summary:")
        print(ds)

        if "latitude" not in ds.coords:
            raise ValueError("Dataset does not contain coordinate 'latitude'")
        ds = ds.sel(latitude=slice(-88.5, 90))  # crop south pole

        if "time" not in ds.coords:
            raise ValueError("Dataset does not contain coordinate 'time'")

        print(f"Dataset loaded with time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        print(f"Dataset variables count: {len(ds.data_vars)}")
        print(f"First 20 variables: {list(ds.data_vars)[:20]}")

        missing = [c for c in channel_names if c not in ds.data_vars]
        if missing:
            print("Available variables:", list(ds.data_vars))
            raise ValueError(
                f"Missing channels in merged dataset, first few: {missing[:20]}"
            )

    repo_name = args.repo_name
    model_name = args.model_name
    print(f"Loading model from {repo_name}/{model_name}")
    vae = AutoencoderDC.from_pretrained(repo_name, subfolder=model_name)

    with open(args.normalization_json) as f:
        normalization_param_dict = json.load(f)

    print(f"Loading static conditioning data from {args.lsm_path} and {args.orography_path}")
    lsm_tensor = torch.load(args.lsm_path, weights_only=True)
    lsm_tensor = lsm_tensor[1:, :]

    orography_tensor = torch.load(args.orography_path, weights_only=True)
    orography_tensor = orography_tensor[:, 1:, :]

    static_conditioning_tensor = lsm_tensor.unsqueeze(0)
    static_conditioning_tensor = torch.cat([static_conditioning_tensor, orography_tensor], dim=0)

    static_mean_tensor = static_conditioning_tensor.mean(dim=(1, 2), keepdim=True)
    static_std_tensor = static_conditioning_tensor.std(dim=(1, 2), keepdim=True)
    static_conditioning_tensor = (
                                         static_conditioning_tensor - static_mean_tensor
                                 ) / static_std_tensor

    print(f"Encoding data using {args.method} method and saving to {zarr_output_path}")

    if args.method == "direct":
        encode_latents_and_save_zarr_direct(
            ds=ds,
            vae=vae,
            zarr_path=zarr_output_path,
            channel_names=channel_names,
            normalization_param_dict=normalization_param_dict,
            static_conditioning_tensor=static_conditioning_tensor,
        )
    else:
        if not start_date or not end_date:
            raise ValueError("start_date and end_date must be provided for HF dataset method")

        encode_latents_and_save_zarr_hf_dataset(
            zarr_path=zarr_output_path,
            vae=vae,
            channel_names=channel_names,
            normalization_param_dict=normalization_param_dict,
            static_conditioning_tensor=static_conditioning_tensor,
            start_date=start_date,
            end_date=end_date,
            dataset_py_path=args.dataset_py_path,
        )


if __name__ == "__main__":
    main()