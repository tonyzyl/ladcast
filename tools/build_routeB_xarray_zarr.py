import os
import shutil
import argparse
import zarr
import numpy as np
import pandas as pd
import xarray as xr


def build_block(
        p_data,
        s_data,
        p_idx,
        s_idx,
        levels,
        lat,
        lon,
        times_full,
        g0,
        g1,
):
    """
    构建全局时间区间 [g0, g1) 对应的数据块。
    注意：g0 必须 >= 5，因为 tp6hr 需要前 5 个时刻做严格 6h rolling。
    """
    pressure_map = {
        "z": "geopotential",
        "q": "specific_humidity",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
    }

    surface_map = {
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "t2m": "2m_temperature",
        "msl": "mean_sea_level_pressure",
    }

    times_block = times_full[g0:g1]
    data_vars = {}

    # pressure vars
    for raw_name, std_name in pressure_map.items():
        arr = np.asarray(p_data[g0:g1, p_idx[raw_name], :, :, :], dtype=np.float32)
        data_vars[std_name] = xr.DataArray(
            arr,
            dims=("time", "level", "latitude", "longitude"),
            coords={
                "time": times_block,
                "level": levels,
                "latitude": lat,
                "longitude": lon,
            },
        )

    # surface vars
    for raw_name, std_name in surface_map.items():
        arr = np.asarray(s_data[g0:g1, s_idx[raw_name], :, :], dtype=np.float32)
        data_vars[std_name] = xr.DataArray(
            arr,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": times_block,
                "latitude": lat,
                "longitude": lon,
            },
        )

    # strict tp6hr: read overlap [g0-5, g1)
    tp_overlap = np.asarray(s_data[g0 - 5:g1, s_idx["tp"], :, :], dtype=np.float32)
    tp_overlap_da = xr.DataArray(
        tp_overlap,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": times_full[g0 - 5:g1],
            "latitude": lat,
            "longitude": lon,
        },
    )

    tp6_overlap = tp_overlap_da.rolling(time=6, min_periods=6).sum()
    tp6_block = tp6_overlap.isel(time=slice(5, None))
    tp6_block = tp6_block.where(tp6_block >= 0, 0)
    tp6_block = tp6_block.rename("total_precipitation_6hr")

    # 对齐当前块时间坐标
    tp6_block = tp6_block.assign_coords(time=times_block)
    data_vars["total_precipitation_6hr"] = tp6_block

    ds_block = xr.Dataset(data_vars)
    return ds_block


def main():
    parser = argparse.ArgumentParser(description="Blockwise build routeB xarray zarr from raw ERA5 zarr.")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/data_large/zarr_datasets/ERA5_1_5_1h_zarr_conservative_1979-2024",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="/data_large/zarr_datasets/ERA5_ladcast_routeB_1979_2024.zarr",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default="1979-01-01 00:00:00",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=None,
        help="仅处理前 n 个时间步；默认全量。",
    )
    parser.add_argument(
        "--block-hours",
        type=int,
        default=48,
        help="每次处理多少小时。建议 24/48/72。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    print("==== Open raw zarr ====")
    p_group = zarr.open_group(f"{args.raw_root}/pressure", mode="r")
    s_group = zarr.open_group(f"{args.raw_root}/surface", mode="r")

    p_data = p_group["data"]   # (time, var, level, lat, lon)
    s_data = s_group["data"]   # (time, var, lat, lon)

    p_vars = list(p_group["var"][:])
    s_vars = list(s_group["var"][:])
    levels = np.array(p_group["level"][:])
    lat = np.array(p_group["latitude"][:])
    lon = np.array(p_group["longitude"][:])
    p_time_idx = np.array(p_group["time"][:])
    s_time_idx = np.array(s_group["time"][:])

    assert p_data.shape[0] == s_data.shape[0], "pressure/surface time dim mismatch"
    assert np.array_equal(p_time_idx, s_time_idx), "pressure/surface time index mismatch"

    p_idx = {v: i for i, v in enumerate(p_vars)}
    s_idx = {v: i for i, v in enumerate(s_vars)}

    for v in ["z", "q", "t", "u", "v"]:
        if v not in p_idx:
            raise KeyError(f"Missing pressure variable: {v}")
    for v in ["u10", "v10", "t2m", "msl", "tp"]:
        if v not in s_idx:
            raise KeyError(f"Missing surface variable: {v}")

    total_time = len(p_time_idx)
    n_time = total_time if args.n_time is None else min(args.n_time, total_time)

    # 全局时间坐标
    times_full = pd.date_range(args.start_time, periods=n_time, freq="1h")

    # 因为严格 tp6hr，需要丢掉前 5 个时间步
    global_start = 5
    global_end = n_time

    if global_end <= global_start:
        raise ValueError("n_time 太小，无法构建严格 6h 累计降水。至少需要 6 个时间步。")

    if os.path.exists(args.out_path):
        if args.overwrite:
            print(f"Removing existing output: {args.out_path}")
            shutil.rmtree(args.out_path)
        else:
            raise FileExistsError(f"{args.out_path} exists. Use --overwrite.")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print("==== Start blockwise writing ====")
    print(f"raw total time     : {total_time}")
    print(f"selected n_time    : {n_time}")
    print(f"effective time span: {global_end - global_start}")
    print(f"block_hours        : {args.block_hours}")
    print(f"output path        : {args.out_path}")

    first = True
    block_id = 0

    # 编码时固定 on-disk chunk
    encoding = {
        "geopotential": {"chunks": (24, len(levels), len(lat), len(lon))},
        "specific_humidity": {"chunks": (24, len(levels), len(lat), len(lon))},
        "temperature": {"chunks": (24, len(levels), len(lat), len(lon))},
        "u_component_of_wind": {"chunks": (24, len(levels), len(lat), len(lon))},
        "v_component_of_wind": {"chunks": (24, len(levels), len(lat), len(lon))},
        "10m_u_component_of_wind": {"chunks": (24, len(lat), len(lon))},
        "10m_v_component_of_wind": {"chunks": (24, len(lat), len(lon))},
        "2m_temperature": {"chunks": (24, len(lat), len(lon))},
        "mean_sea_level_pressure": {"chunks": (24, len(lat), len(lon))},
        "total_precipitation_6hr": {"chunks": (24, len(lat), len(lon))},
    }

    for g0 in range(global_start, global_end, args.block_hours):
        g1 = min(g0 + args.block_hours, global_end)
        block_id += 1
        print(f"\n[Block {block_id}] global time range = [{g0}, {g1})  len={g1-g0}")

        ds_block = build_block(
            p_data=p_data,
            s_data=s_data,
            p_idx=p_idx,
            s_idx=s_idx,
            levels=levels,
            lat=lat,
            lon=lon,
            times_full=times_full,
            g0=g0,
            g1=g1,
        )

        if first:
            ds_block.to_zarr(
                args.out_path,
                mode="w",
                consolidated=False,
                zarr_format=2,
                encoding=encoding,
            )
            first = False
        else:
            ds_block.to_zarr(
                args.out_path,
                mode="a",
                append_dim="time",
                consolidated=False,
            )

        print(
            f"  block saved: {str(ds_block.time.values[0])} -> {str(ds_block.time.values[-1])}"
        )

    print("\n==== Consolidate metadata ====")
    zarr.consolidate_metadata(args.out_path)

    print("==== Done ====")
    print(f"saved to: {args.out_path}")


if __name__ == "__main__":
    main()