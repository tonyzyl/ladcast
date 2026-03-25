import argparse
import os

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RouteB latent zarr statistics.")
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latent_path = os.path.expanduser(args.latent_path)

    ds = xr.open_zarr(latent_path, consolidated=False)
    if "latent" not in ds:
        raise KeyError("Expected variable 'latent' in zarr dataset.")

    latent = ds["latent"]  # (time, channel, lat, lon)
    if "time" not in latent.dims or "channel" not in latent.dims:
        raise RuntimeError(f"Unexpected dims for latent: {latent.dims}")

    time_len = int(latent.sizes["time"])
    time_vals = ds["time"].values
    time_start = str(time_vals[0]) if time_len > 0 else "N/A"
    time_end = str(time_vals[-1]) if time_len > 0 else "N/A"

    print("===== RouteB Latent Check =====")
    print(f"time length: {time_len}")
    print(f"time range: {time_start} -> {time_end}")
    print(f"latent variable shape: {tuple(latent.shape)}")
    print(f"dtype: {latent.dtype}")
    print(f"chunks: {latent.chunks}")

    if time_len == 0:
        raise RuntimeError("Latent dataset has no time samples.")

    if args.max_batches is not None:
        # batch mode by chunks along time dimension if present
        chunk_t = latent.chunks[0][0] if latent.chunks and latent.chunks[0] else 1
        max_samples = min(time_len, int(chunk_t * args.max_batches))
    else:
        max_samples = min(time_len, int(args.max_samples))

    arr = latent.isel(time=slice(0, max_samples)).values

    finite_mask = np.isfinite(arr)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    print(f"checked samples: {max_samples}")
    print(f"has_nan: {bool(has_nan)}")
    print(f"has_inf: {bool(has_inf)}")

    if not finite_mask.all():
        bad_count = int((~finite_mask).sum())
        print(f"non_finite_count: {bad_count}")

    # per-channel stats across (time, lat, lon)
    ch_mean = arr.mean(axis=(0, 2, 3))
    ch_std = arr.std(axis=(0, 2, 3))
    ch_min = arr.min(axis=(0, 2, 3))
    ch_max = arr.max(axis=(0, 2, 3))

    print("\nPer-channel stats:")
    for c in range(arr.shape[1]):
        print(
            f"- channel {c:03d}: mean={ch_mean[c]:.8f} std={ch_std[c]:.8f} "
            f"min={ch_min[c]:.8f} max={ch_max[c]:.8f}"
        )


if __name__ == "__main__":
    main()