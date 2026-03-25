import os
import json
import argparse
import numpy as np
import xarray as xr


PRESSURE_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

SURFACE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]


def to_python_float(x):
    return float(np.asarray(x).item())


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization stats for routeB ERA5 dataset."
    )
    parser.add_argument(
        "--ds-path",
        type=str,
        default="~/data/ERA5_ladcast_routeB_1979_2024.zarr",
        help="Path to routeB xarray zarr dataset.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="~/ladcast/ladcast/static/ERA5_routeB_normal_1979_2017.json",
        help="Output normalization json path.",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="1979-01-01",
        help="Training start time.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2017-12-31T23:00:00",
        help="Training end time.",
    )
    parser.add_argument(
        "--std-eps",
        type=float,
        default=1e-6,
        help="Minimum std epsilon to avoid zero std.",
    )
    args = parser.parse_args()

    ds_path = os.path.expanduser(args.ds_path)
    out_json = os.path.expanduser(args.out_json)

    print("==== Open dataset ====")
    print(f"dataset path: {ds_path}")
    ds = xr.open_zarr(ds_path, consolidated=True)

    print("==== Select training range ====")
    train_ds = ds.sel(time=slice(args.train_start, args.train_end))
    print(train_ds)
    print("train time length:", train_ds.sizes["time"])
    print("train time range:", train_ds.time.values[0], "->", train_ds.time.values[-1])

    stats = {}

    print("\n==== Compute pressure variable stats (per level) ====")
    for var in PRESSURE_VARS:
        if var not in train_ds:
            raise KeyError(f"Missing pressure variable in dataset: {var}")

        da = train_ds[var]

        # mean/std over (time, latitude, longitude), keep level
        mean = da.mean(dim=("time", "latitude", "longitude")).compute().values
        std = da.std(dim=("time", "latitude", "longitude")).compute().values

        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)

        std = np.where(std < args.std_eps, args.std_eps, std)

        stats[var] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        print(f"\n{var}")
        print("  mean shape:", mean.shape)
        print("  std shape :", std.shape)
        print("  mean[0:3] :", mean[:3].tolist())
        print("  std[0:3]  :", std[:3].tolist())
        print("  min(std)  :", float(std.min()))
        print("  max(std)  :", float(std.max()))

    print("\n==== Compute surface variable stats (scalar) ====")
    for var in SURFACE_VARS:
        if var not in train_ds:
            raise KeyError(f"Missing surface variable in dataset: {var}")

        da = train_ds[var]

        mean = to_python_float(da.mean().compute().values)
        std = to_python_float(da.std().compute().values)

        if std < args.std_eps:
            std = args.std_eps

        stats[var] = {
            "mean": mean,
            "std": std,
        }

        print(f"\n{var}")
        print("  mean =", mean)
        print("  std  =", std)

    print("\n==== Save json ====")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"saved to: {out_json}")


if __name__ == "__main__":
    main()