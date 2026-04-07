#!/usr/bin/env python3
"""
Pinpoint exactly where the DataLoader hangs.
Each step has a 60s timeout.

Usage:
    python tools/diagnose_dataloader.py --config your_config.yaml
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("TIMEOUT")

def timed(label, func, timeout_sec=60):
    """Run func() with a timeout. Returns (result, elapsed) or raises."""
    print(f"  [{label}] ...", end="", flush=True)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        signal.alarm(0)
        print(f" OK ({elapsed:.1f}s)")
        return result, elapsed
    except TimeoutError:
        print(f" ✗ TIMED OUT after {timeout_sec}s ← HANG IS HERE")
        signal.alarm(0)
        return None, timeout_sec
    except Exception as e:
        signal.alarm(0)
        elapsed = time.time() - t0
        print(f" ✗ ERROR ({elapsed:.1f}s): {e}")
        return None, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per step in seconds")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    dl_cfg = OmegaConf.to_container(config.get("train_dataloader", {}), resolve=True)

    ds_path = dl_cfg.get("ds_path", "")
    xr_engine = dl_cfg.get("xr_engine", "zarr")
    start_date = dl_cfg.get("start_date", "1979-01-01")
    end_date = dl_cfg.get("end_date", "2017-12-31")
    var_name = dl_cfg.get("var_name", "latents")

    print("=" * 60)
    print("DataLoader Hang Diagnosis (each step has timeout)")
    print("=" * 60)
    print(f"  ds_path:    {ds_path}")
    print(f"  engine:     {xr_engine}")
    print(f"  date range: {start_date} → {end_date}")
    print(f"  var_name:   {var_name}")
    print(f"  timeout:    {args.timeout}s per step")
    print()

    # Import normalize helper
    from ladcast.dataloader.ar_dataloder import _normalize_zarr_dataset

    # Step 0: Check path exists
    print("Step 0: Check dataset path")
    if os.path.exists(ds_path):
        # Check if it's a zarr store (directory with .zmetadata or .zarray)
        if os.path.isdir(ds_path):
            contents = os.listdir(ds_path)[:20]
            print(f"  Path exists, is directory, contents: {contents}")
        else:
            size = os.path.getsize(ds_path)
            print(f"  Path exists, is file, size: {size / 1024**3:.2f} GB")
    else:
        print(f"  ✗ Path does not exist: {ds_path}")
        print("  Check your ds_path in the config!")
        return

    # Step 1: xr.open_dataset without dask + normalize names
    print("\nStep 1: Open dataset WITHOUT dask (chunks=None) + normalize names")
    import xarray as xr
    ds_nodask, _ = timed("open_dataset(chunks=None)",
        lambda: _normalize_zarr_dataset(xr.open_dataset(ds_path, engine=xr_engine, chunks=None)),
        timeout_sec=args.timeout)
    if ds_nodask is not None:
        print(f"    Variables: {list(ds_nodask.data_vars)}")
        print(f"    Dims: {dict(ds_nodask.sizes)}")
        # After normalize, var_name should be "latents"
        _actual_var = "latents" if "latents" in ds_nodask else var_name
        if _actual_var in ds_nodask:
            v = ds_nodask[_actual_var]
            print(f"    '{_actual_var}' shape: {v.shape}, dtype: {v.dtype}")
            nbytes = v.nbytes
            print(f"    '{_actual_var}' size: {nbytes / 1024**3:.2f} GB")
        else:
            print(f"    ✗ Variable not found! Available: {list(ds_nodask.data_vars)}")
        ds_nodask.close()

    # Step 2: xr.open_dataset WITH dask (chunks="auto") + normalize
    print("\nStep 2: Open dataset WITH dask (chunks='auto') + normalize")
    ds_dask, _ = timed("open_dataset(chunks='auto')",
        lambda: _normalize_zarr_dataset(xr.open_dataset(ds_path, engine=xr_engine, chunks="auto")),
        timeout_sec=args.timeout)

    if ds_dask is None:
        print("  ⚠️  Dask chunking hangs! This is the problem.")
        print("  Fix: Open without dask or specify explicit chunks.")
        return

    # Step 3: .sel(time=slice(...))
    print("\nStep 3: Time selection .sel()")
    ds_sel, _ = timed("sel(time=slice)",
        lambda: ds_dask.sel(time=slice(start_date, end_date)),
        timeout_sec=args.timeout)

    if ds_sel is None:
        print("  ⚠️  Time selection hangs!")
        return

    # Step 4: .transpose()
    print("\nStep 4: Transpose")
    ds_t, _ = timed("transpose('C','time','H','W')",
        lambda: ds_sel.transpose("C", "time", "H", "W"),
        timeout_sec=args.timeout)

    if ds_t is None:
        print("  ⚠️  Transpose hangs!")
        return

    # Step 5: Access .shape
    print("\nStep 5: Access data shape")
    def get_shape():
        da = ds_t["latents"]
        return da.shape
    shape, _ = timed("data.shape", get_shape, timeout_sec=args.timeout)
    if shape is not None:
        print(f"    Shape: {shape}")

    # Step 6: isel (like XarrayDataset3D.__init__)
    print("\nStep 6: isel (time slicing)")
    def do_isel():
        da = ds_t["latents"]
        return da.isel(time=slice(0, None, 1))
    da_isel, _ = timed("isel(time=slice(0,None,1))", do_isel, timeout_sec=args.timeout)

    # Step 7: Actually read one sample (the critical test)
    print("\nStep 7: Read ONE sample (.to_numpy())")
    def read_one():
        da = ds_t["latents"]
        sample = da[:, 0:2]  # (C, 2, H, W)
        return sample.to_numpy()
    arr, elapsed = timed("data[:, 0:2].to_numpy()", read_one, timeout_sec=args.timeout)
    if arr is not None:
        print(f"    Result shape: {arr.shape}, dtype: {arr.dtype}")
        if elapsed > 5:
            print(f"  ⚠️  Single sample took {elapsed:.1f}s — very slow!")
            print(f"    At this rate, 1 epoch ≈ {elapsed * shape[1] / 3600:.1f} hours just for data loading")

    # Step 8: Read 10 samples in sequence
    print("\nStep 8: Read 10 sequential samples")
    def read_ten():
        da = ds_t["latents"]
        results = []
        for i in range(10):
            s = da[:, i:i+2].to_numpy()
            results.append(s)
        return results
    res, elapsed = timed("10 x data[:, i:i+2].to_numpy()", read_ten, timeout_sec=args.timeout)
    if res is not None:
        print(f"    Avg per sample: {elapsed/10:.2f}s")

    # Step 9: Try with synchronous dask scheduler
    print("\nStep 9: Read with dask synchronous scheduler")
    import dask
    def read_sync():
        with dask.config.set(scheduler='synchronous'):
            da = ds_t["latents"]
            return da[:, 0:2].to_numpy()
    arr2, _ = timed("synchronous scheduler", read_sync, timeout_sec=args.timeout)

    # Step 10: Read WITHOUT dask at all
    print("\nStep 10: Read WITHOUT dask (chunks=None)")
    def read_nodask():
        ds2 = _normalize_zarr_dataset(xr.open_dataset(ds_path, engine=xr_engine, chunks=None))
        ds2 = ds2.sel(time=slice(start_date, end_date))
        ds2 = ds2.transpose("C", "time", "H", "W")
        da = ds2["latents"]
        return da[:, 0:2].values
    arr3, elapsed_nodask = timed("no-dask read", read_nodask, timeout_sec=args.timeout)

    # Step 11: Full DataLoader test (num_workers=0, 5 batches)
    print("\nStep 11: Full DataLoader (5 batches, num_workers=0)")
    latent_norm_path = "ladcast/static/ERA5_latent_normal_1979_2017_lat84.json"
    if os.path.exists(latent_norm_path):
        with open(latent_norm_path, "r") as f:
            dl_cfg["transform_args"] = json.load(f)
            dl_cfg["transform_args"]["target_std"] = 0.5
    dl_cfg["num_workers"] = 0
    dl_cfg["persistent_workers"] = False
    dl_cfg["prefetch_factor"] = None

    from ladcast.dataloader.ar_dataloder import prepare_ar_dataloader
    def run_dl():
        dl = prepare_ar_dataloader(**dl_cfg)
        for i, batch in enumerate(dl):
            if i >= 4:
                break
        return len(dl)
    n_batches, _ = timed("DataLoader 5 batches", run_dl, timeout_sec=args.timeout * 2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If any step above shows TIMED OUT, that's where the hang is.")
    print()
    print("Common fixes:")
    print("  - If Step 2 hangs: add to config or training script:")
    print("      xr.open_dataset(path, engine='zarr', chunks=None)")
    print("    instead of chunks='auto'")
    print("  - If Step 7/8 hangs (dask read): use synchronous dask scheduler:")
    print("      import dask; dask.config.set(scheduler='synchronous')")
    print("  - If Step 10 is fast but Step 7 hangs: dask is the problem,")
    print("      open with chunks=None in ar_dataloder.py")
    print("  - If everything hangs: storage system issue (NFS/GPFS/disk)")


if __name__ == "__main__":
    main()
