import argparse
import json
import os

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RouteB latent per-channel mean/std.")
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--start_time", type=str, default="1979-01-01")
    parser.add_argument("--end_time", type=str, default="2017-12-31")
    parser.add_argument(
        "--output_json",
        type=str,
        default="~/ladcast/static/ERA5_routeB_latent_normal_1979_2017.json",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = xr.open_zarr(os.path.expanduser(args.latent_path), consolidated=False)
    if "latent" not in ds:
        raise KeyError("Expected variable 'latent' in zarr dataset.")

    latent = ds["latent"].sel(time=slice(args.start_time, args.end_time))
    if latent.sizes.get("time", 0) <= 0:
        raise RuntimeError("No samples found in selected training time range.")

    arr = latent.values  # (T, C, H, W)
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite values found in latent data during norm computation.")

    mean = arr.mean(axis=(0, 2, 3)).astype(np.float64)
    std = arr.std(axis=(0, 2, 3)).astype(np.float64)
    std = np.maximum(std, args.eps)

    out = {
        "start_time": args.start_time,
        "end_time": args.end_time,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    output_json = os.path.expanduser(args.output_json)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    mean_min, mean_max = float(mean.min()), float(mean.max())
    std_min, std_max = float(std.min()), float(std.max())

    print("===== RouteB Latent Norm Computed =====")
    print(f"samples: {arr.shape[0]}")
    print(f"channels: {arr.shape[1]}")
    print(f"mean_range: [{mean_min:.8f}, {mean_max:.8f}]")
    print(f"std_range:  [{std_min:.8f}, {std_max:.8f}]")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()