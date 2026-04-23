#!/usr/bin/env python3
"""
Minimal test: step through prepare_ar_dataloader line by line.
Usage: python tools/test_dataloader_minimal.py --config your_config.yaml
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
 
    from omegaconf import OmegaConf
    import xarray as xr
    import torch
 
    config = OmegaConf.load(args.config)
    dl_cfg = OmegaConf.to_container(config.get("train_dataloader", {}), resolve=True)
 
    ds_path = dl_cfg["ds_path"]
    xr_engine = dl_cfg.get("xr_engine", "zarr")
    start_date = dl_cfg.get("start_date", "1979-01-01")
    end_date = dl_cfg.get("end_date", "2017-12-31")
    var_name = dl_cfg.get("var_name", "latents")
 
    from ladcast.dataloader.ar_dataloder import _normalize_zarr_dataset
 
    # Step A: Open
    print(f"A) xr.open_dataset(chunks=None) ...", flush=True)
    t0 = time.time()
    xarr = xr.open_dataset(ds_path, engine=xr_engine, chunks=None)
    print(f"   OK ({time.time()-t0:.1f}s), dims={dict(xarr.sizes)}, vars={list(xarr.data_vars)}")
 
    # Step B: Normalize names
    print(f"B) _normalize_zarr_dataset ...", flush=True)
    t0 = time.time()
    xarr = _normalize_zarr_dataset(xarr)
    print(f"   OK ({time.time()-t0:.1f}s), dims={dict(xarr.sizes)}, vars={list(xarr.data_vars)}")
 
    # Step C: sel
    print(f"C) .sel(time=slice('{start_date}','{end_date}')) ...", flush=True)
    t0 = time.time()
    xarr = xarr.sel(time=slice(start_date, end_date))
    print(f"   OK ({time.time()-t0:.1f}s), time dim={xarr.sizes['time']}")
 
    # Step D: transpose
    print(f"D) .transpose('C','time','H','W') ...", flush=True)
    t0 = time.time()
    xarr = xarr.transpose("C", "time", "H", "W")
    print(f"   OK ({time.time()-t0:.1f}s)")
 
    # Step E: access var
    print(f"E) xarr['{var_name}'] ...", flush=True)
    t0 = time.time()
    try:
        da = xarr[var_name]
        print(f"   OK ({time.time()-t0:.1f}s), shape={da.shape}, dtype={da.dtype}")
    except KeyError:
        print(f"   ✗ KeyError! '{var_name}' not found. Available: {list(xarr.data_vars)}")
        print(f"   Fix: set var_name to one of the above in your config")
        return
 
    # Step F: XarrayDataset3D init
    print(f"F) XarrayDataset3D.__init__ ...", flush=True)
    t0 = time.time()
    from ladcast.dataloader.ar_dataloder import XarrayDataset3D
    # Load transform_args
    latent_norm_path = "static/ERA5_routeB_latent_normal_1979_2017.json"
    transform_args = None
    if os.path.exists(latent_norm_path):
        with open(latent_norm_path, "r") as f:
            transform_args = json.load(f)
            transform_args["target_std"] = 0.5
 
    input_seq_len = dl_cfg.get("input_seq_len", 1)
    return_seq_len = dl_cfg.get("return_seq_len", 1)
 
    dataset = XarrayDataset3D(
        data=da,
        transform=dl_cfg.get("transform", "normalize"),
        transform_args=transform_args,
        input_seq_len=input_seq_len,
        return_seq_len=return_seq_len,
        truncate_first=dl_cfg.get("truncate_first", 0),
        sampling_interval=dl_cfg.get("sampling_interval", 1),
        interval_between_pred=dl_cfg.get("interval_between_pred", 1),
        data_augmentation=dl_cfg.get("data_augmentation", False),
        load_in_memory=dl_cfg.get("load_in_memory", False),
    )
    print(f"   OK ({time.time()-t0:.1f}s), len={len(dataset)}")
 
    # Step G: Single __getitem__
    print(f"G) dataset[0] ...", flush=True)
    t0 = time.time()
    sample = dataset[0]
    print(f"   OK ({time.time()-t0:.1f}s)")
    print(f"   input shape: {sample[0].shape}, target shape: {sample[1].shape}, ts: {sample[2]}")
 
    # Step H: 5 more __getitem__
    print(f"H) dataset[1] to dataset[5] ...", flush=True)
    for i in range(1, 6):
        t0 = time.time()
        s = dataset[i]
        print(f"   [{i}] OK ({time.time()-t0:.1f}s)")
 
    # Step I: DataLoader wrapping
    print(f"I) DataLoader(batch_size={dl_cfg.get('batch_size',1)}, num_workers=0) ...", flush=True)
    from torch.utils.data import DataLoader
    t0 = time.time()
    dl = DataLoader(dataset, batch_size=dl_cfg.get("batch_size", 1), shuffle=False, num_workers=0)
    print(f"   OK ({time.time()-t0:.1f}s)")
 
    # Step J: iterate 3 batches
    print(f"J) Iterating 3 batches ...", flush=True)
    for i, batch in enumerate(dl):
        print(f"   batch {i}: input={batch[0].shape}, target={batch[1].shape} ({time.time()-t0:.1f}s)")
        if i >= 2:
            break
 
    print("\n✓ ALL PASSED - DataLoader works correctly!")
 
if __name__ == "__main__":
    main()
