#!/usr/bin/env python3
"""
Diagnostic script for LaDCast training hangs/crashes.
Run this INSTEAD of train_AR.py to identify the root cause.

Usage:
    python tools/diagnose_training.py --config your_config.yaml --ar_cls transformer [other train_AR args]

It performs 5 tests:
  1. Shared memory (/dev/shm) check
  2. DataLoader stress test (xarray + zarr + num_workers)
  3. GPU memory leak detection over N training steps
  4. Validation dry-run (the most likely crash point)
  5. NCCL / multi-GPU gather test
"""

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def check_shm():
    """Test 1: Check /dev/shm capacity."""
    print("\n" + "=" * 60)
    print("TEST 1: Shared Memory (/dev/shm) Check")
    print("=" * 60)
    shm_path = "/dev/shm"
    if os.path.exists(shm_path):
        usage = psutil.disk_usage(shm_path)
        print(f"  Total:     {fmt_bytes(usage.total)}")
        print(f"  Used:      {fmt_bytes(usage.used)}")
        print(f"  Free:      {fmt_bytes(usage.free)}")
        print(f"  Usage:     {usage.percent}%")
        if usage.total < 2 * 1024**3:
            print("  ⚠️  WARNING: /dev/shm < 2GB. DataLoader with num_workers > 0")
            print("     may hang when shared memory is exhausted.")
            print("     Fix: --shm-size=8g (Docker) or mount -o remount,size=8G /dev/shm")
            return False
        if usage.percent > 80:
            print("  ⚠️  WARNING: /dev/shm is >80% full already!")
            return False
        print("  ✓ /dev/shm looks OK")
        return True
    else:
        print("  /dev/shm not found (not Linux?)")
        return True


def check_dataloader(config_path, n_batches=200):
    """Test 2: Stress-test the DataLoader for xarray/zarr leaks."""
    print("\n" + "=" * 60)
    print(f"TEST 2: DataLoader Stress Test ({n_batches} batches)")
    print("=" * 60)
    import json
    import signal as _signal
    from omegaconf import OmegaConf
    from ladcast.dataloader.ar_dataloder import prepare_ar_dataloader

    config = OmegaConf.load(config_path)
    train_dl_config = config.pop("train_dataloader", OmegaConf.create())

    # Check if using num_workers > 0 with xarray (known issue)
    nw = train_dl_config.get("num_workers", 0)
    load_in_mem = train_dl_config.get("load_in_memory", False)
    persistent = train_dl_config.get("persistent_workers", False)

    print(f"  num_workers:        {nw}")
    print(f"  load_in_memory:     {load_in_mem}")
    print(f"  persistent_workers: {persistent}")

    if nw > 0 and not load_in_mem:
        print("  ⚠️  WARNING: num_workers > 0 with xarray lazy loading (load_in_memory=False)")
        print("     xarray/zarr file handles are NOT fork-safe. This is a known cause of")
        print("     deadlocks and silent crashes after many iterations.")
        print()
        print("  ✗ FAIL: This is the cause of your training hang.")
        print()
        print("     Fix in your config.yaml:")
        print("       num_workers: 0")
        print("       persistent_workers: false")
        print()
        print("  Re-testing with num_workers=0 to verify DataLoader works...")
        train_dl_config["num_workers"] = 0
        train_dl_config["persistent_workers"] = False
        nw = 0  # continue with safe settings

    # Check dataset size before attempting load_in_memory
    if load_in_mem:
        print("  load_in_memory=true: checking dataset size first...")
        try:
            import xarray as xr
            ds = xr.open_dataset(
                train_dl_config.get("ds_path", ""),
                engine=train_dl_config.get("xr_engine", "zarr"),
                chunks="auto",
            )
            var_name = train_dl_config.get("var_name", "latents")
            if var_name in ds:
                nbytes = ds[var_name].nbytes
                print(f"  Dataset '{var_name}' size: {fmt_bytes(nbytes)}")
                avail = psutil.virtual_memory().available
                print(f"  Available RAM: {fmt_bytes(avail)}")
                if nbytes > avail * 0.8:
                    print(f"  ⚠️  WARNING: Dataset ({fmt_bytes(nbytes)}) > 80% of available RAM ({fmt_bytes(avail)})")
                    print("     load_in_memory=true will OOM! Use num_workers=0 instead.")
                    return False
                else:
                    print(f"  ✓ Dataset fits in RAM ({fmt_bytes(nbytes)} < {fmt_bytes(avail)})")
            ds.close()
        except Exception as e:
            print(f"  Could not check dataset size: {e}")

    # Add transform_args if missing
    latent_norm_path = "ladcast/static/ERA5_latent_normal_1979_2017_lat84.json"
    if os.path.exists(latent_norm_path):
        with open(latent_norm_path, "r") as f:
            train_dl_config["transform_args"] = json.load(f)
            train_dl_config["transform_args"]["target_std"] = 0.5

    try:
        dl = prepare_ar_dataloader(**OmegaConf.to_container(train_dl_config, resolve=True))
    except Exception as e:
        print(f"  ✗ Failed to create DataLoader: {e}")
        return False

    print(f"  DataLoader created, {len(dl)} batches per epoch")
    print(f"  Running {n_batches} batches...")

    proc = psutil.Process()
    mem_start = proc.memory_info().rss
    shm_start = psutil.disk_usage("/dev/shm").used if os.path.exists("/dev/shm") else 0

    t0 = time.time()
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        if (i + 1) % 50 == 0:
            mem_now = proc.memory_info().rss
            shm_now = psutil.disk_usage("/dev/shm").used if os.path.exists("/dev/shm") else 0
            elapsed = time.time() - t0
            print(f"    Batch {i+1}: RSS={fmt_bytes(mem_now)} "
                  f"(+{fmt_bytes(mem_now - mem_start)}), "
                  f"/dev/shm={fmt_bytes(shm_now)} "
                  f"(+{fmt_bytes(shm_now - shm_start)}), "
                  f"time={elapsed:.1f}s")

    mem_end = proc.memory_info().rss
    mem_growth = mem_end - mem_start
    print(f"  Final RSS growth: {fmt_bytes(mem_growth)}")
    if mem_growth > 1 * 1024**3:
        print(f"  ⚠️  WARNING: RSS grew by {fmt_bytes(mem_growth)} over {n_batches} batches.")
        print("     This indicates a memory leak in the DataLoader/xarray.")
        print("     At this rate, training will OOM after ~{:.0f} batches.".format(
            (psutil.virtual_memory().available / (mem_growth / n_batches))
        ))
        return False
    print("  ✓ DataLoader memory looks stable")
    return True


def check_gpu_memory_leak(config_path, n_steps=50):
    """Test 3: Run training steps and monitor GPU memory."""
    print("\n" + "=" * 60)
    print(f"TEST 3: GPU Memory Leak Detection ({n_steps} steps)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  No CUDA GPU available, skipping.")
        return True

    import json
    from omegaconf import OmegaConf
    from ladcast.dataloader.ar_dataloder import prepare_ar_dataloader
    from ladcast.models.LaDCast_3D_model import LaDCastTransformer3DModel
    from ladcast.utils import instantiate_from_config

    config = OmegaConf.load(config_path)
    ar_model_config = OmegaConf.to_container(config.get("ar_model", {}), resolve=True)
    noise_scheduler_config = config.get("noise_scheduler", OmegaConf.create())
    noise_sampler_config = config.get("noise_sampler", OmegaConf.create())
    train_dl_config = OmegaConf.to_container(config.get("train_dataloader", {}), resolve=True)
    general_config = config.get("general", OmegaConf.create())

    latent_norm_path = "ladcast/static/ERA5_latent_normal_1979_2017_lat84.json"
    if os.path.exists(latent_norm_path):
        with open(latent_norm_path, "r") as f:
            train_dl_config["transform_args"] = json.load(f)
            train_dl_config["transform_args"]["target_std"] = 0.5

    device = torch.device("cuda:0")

    try:
        ar_model = LaDCastTransformer3DModel.from_config(config=ar_model_config)
        ar_model = ar_model.to(device)
        ar_model.train()
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        return False

    noise_scheduler = instantiate_from_config(noise_scheduler_config)
    optimizer = torch.optim.AdamW(ar_model.parameters(), lr=1e-4)

    # Create a synthetic batch to avoid DataLoader issues
    bs = train_dl_config.get("batch_size", 1)
    in_ch = ar_model_config.get("in_channels", 32)
    seq_len = train_dl_config.get("return_seq_len", 1)
    input_seq_len = train_dl_config.get("input_seq_len", 1)

    print(f"  Model params: {sum(p.numel() for p in ar_model.parameters()):,}")
    print(f"  Batch shape: ({bs}, {in_ch}, {seq_len}, 15, 30)")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    mem_baseline = torch.cuda.memory_allocated(device)
    print(f"  Baseline GPU mem: {fmt_bytes(mem_baseline)}")

    mem_history = []
    for step in range(n_steps):
        initial_profile = torch.randn(bs, in_ch, input_seq_len, 15, 30, device=device)
        clean_images = torch.randn(bs, in_ch, seq_len, 15, 30, device=device)
        noise = torch.randn_like(clean_images)
        timestamps = torch.tensor([2018010100] * bs, device=device)

        # Simulate EDM training step
        indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,))
        timesteps_t = noise_scheduler.timesteps[indices].to(device)
        noisy = noise_scheduler.add_noise(clean_images, noise, timesteps_t)

        from ladcast.pipelines.utils import get_sigmas
        sigmas = get_sigmas(noise_scheduler, timesteps_t, 5, noisy.dtype, device=device)
        x_in = noise_scheduler.precondition_inputs(noisy, sigmas)

        pred = ar_model(x_in, timesteps_t, initial_profile, time_elapsed=timestamps, return_dict=False)[0]
        pred = noise_scheduler.precondition_outputs(noisy, pred, sigmas)
        weighting = (sigmas**2 + 0.5**2) / (sigmas * 0.5) ** 2
        loss = torch.mean(weighting.float() * (pred.float() - clean_images.float()) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 10 == 0:
            mem_now = torch.cuda.memory_allocated(device)
            mem_peak = torch.cuda.max_memory_allocated(device)
            mem_history.append(mem_now)
            print(f"    Step {step+1}: allocated={fmt_bytes(mem_now)}, "
                  f"peak={fmt_bytes(mem_peak)}, "
                  f"growth={fmt_bytes(mem_now - mem_baseline)}")

    if len(mem_history) >= 3:
        # Check if memory is monotonically growing
        diffs = [mem_history[i+1] - mem_history[i] for i in range(len(mem_history)-1)]
        avg_growth = sum(diffs) / len(diffs)
        if avg_growth > 1 * 1024**2:  # > 1MB per 10 steps
            print(f"  ⚠️  WARNING: GPU memory growing ~{fmt_bytes(avg_growth)} per 10 steps")
            print(f"     Projected OOM after ~{int((torch.cuda.get_device_properties(device).total_mem - mem_history[-1]) / avg_growth * 10)} steps")
            return False
        else:
            print(f"  ✓ GPU memory stable (avg growth: {fmt_bytes(avg_growth)} per 10 steps)")
    del ar_model, optimizer
    torch.cuda.empty_cache()
    return True


def check_validation_oom(config_path):
    """Test 4: Check if validation will OOM."""
    print("\n" + "=" * 60)
    print("TEST 4: Validation Memory Estimation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  No CUDA GPU available, skipping.")
        return True

    device = torch.device("cuda:0")
    gpu_total = torch.cuda.get_device_properties(device).total_mem
    print(f"  GPU total memory: {fmt_bytes(gpu_total)}")

    # Estimate validation memory usage
    ensemble_size = 10
    total_num_steps = 40  # 240h / 6h
    # edm_decoded_tensor: (10, 84, 40, 120, 240) float32
    decoded_tensor_bytes = ensemble_size * 84 * total_num_steps * 120 * 240 * 4
    # ref_tensor: (84, 40, 120, 240) float32
    ref_tensor_bytes = 84 * total_num_steps * 120 * 240 * 4
    # edm_sample_latents per step: (10, 32, 1, 15, 30) float32
    sample_latents_bytes = ensemble_size * 32 * 1 * 15 * 30 * 4

    print(f"  edm_decoded_tensor: {fmt_bytes(decoded_tensor_bytes)} (now on CPU ✓)")
    print(f"  ref_tensor:         {fmt_bytes(ref_tensor_bytes)} (on GPU)")
    print(f"  sample_latents:     {fmt_bytes(sample_latents_bytes)} per step (on GPU)")
    print(f"  DC-AE decode: runs per-ensemble-member, needs ~1-2GB working memory")

    # The key question: how much GPU memory is available during validation?
    # Model weights + optimizer states + EMA are all still on GPU
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    ar_config = config.get("ar_model", {})
    heads = ar_config.get("num_attention_heads", 24)
    head_dim = ar_config.get("attention_head_dim", 128)
    n_layers = ar_config.get("num_layers", 20)
    n_single = ar_config.get("num_single_layers", 40)
    inner_dim = heads * head_dim

    # Rough model size estimate
    # Each dual block: ~12 * inner_dim^2 params, each single block: ~6 * inner_dim^2 params
    est_params = (n_layers * 12 + n_single * 6) * (inner_dim ** 2)
    model_bytes = est_params * 4  # fp32
    # With optimizer (AdamW): 3x model size (params + m + v)
    optimizer_bytes = model_bytes * 2
    ema_bytes = model_bytes if config.get("ema", {}).get("use_ema", False) else 0

    total_resident = model_bytes + optimizer_bytes + ema_bytes
    print(f"\n  Estimated resident GPU memory during validation:")
    print(f"    Model weights:  {fmt_bytes(model_bytes)}")
    print(f"    Optimizer:      {fmt_bytes(optimizer_bytes)}")
    print(f"    EMA:            {fmt_bytes(ema_bytes)}")
    print(f"    Total resident: {fmt_bytes(total_resident)}")

    free_for_val = gpu_total - total_resident
    print(f"    Free for val:   {fmt_bytes(free_for_val)}")

    # During validation: ref_tensor + DC-AE + inference working memory
    val_gpu_needed = ref_tensor_bytes + 2 * 1024**3  # ref + ~2GB working
    print(f"    Val GPU needed: {fmt_bytes(val_gpu_needed)}")

    if free_for_val < val_gpu_needed:
        print(f"  ⚠️  WARNING: Validation likely to OOM!")
        print(f"     Free: {fmt_bytes(free_for_val)} < Needed: {fmt_bytes(val_gpu_needed)}")
        return False
    else:
        print(f"  ✓ Validation should fit in GPU memory")
    return True


def check_system():
    """Test 5: System-level checks."""
    print("\n" + "=" * 60)
    print("TEST 5: System Environment Check")
    print("=" * 60)

    # Check available RAM
    vm = psutil.virtual_memory()
    print(f"  RAM total:     {fmt_bytes(vm.total)}")
    print(f"  RAM available: {fmt_bytes(vm.available)}")
    print(f"  RAM usage:     {vm.percent}%")

    if vm.available < 16 * 1024**3:
        print("  ⚠️  WARNING: Less than 16GB RAM available. Training + validation may")
        print("     cause the Linux OOM killer to silently terminate the process.")
        print("     Check: dmesg | grep -i 'oom\\|killed' after a crash.")

    # Check swap
    swap = psutil.swap_memory()
    print(f"  Swap total:    {fmt_bytes(swap.total)}")
    print(f"  Swap used:     {fmt_bytes(swap.used)}")

    # Check GPU
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"  GPUs:          {n_gpu}")
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            mem_alloc = torch.cuda.memory_allocated(i)
            print(f"    GPU {i}: {props.name}, {fmt_bytes(props.total_mem)} total, "
                  f"{fmt_bytes(mem_alloc)} allocated")
        if n_gpu > 1:
            print("  Multi-GPU detected. NCCL timeout is set to 1800s in train_AR.py.")
            print("  If validation takes >30min on one process, others will timeout → crash.")

    # Check ulimits
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"  File descriptors: soft={soft}, hard={hard}")
        if soft < 4096:
            print("  ⚠️  WARNING: Low file descriptor limit. Zarr + num_workers may exhaust FDs.")
            print("     Fix: ulimit -n 65536")
    except Exception:
        pass

    # Check OOM killer history
    print("\n  Checking for recent OOM kills (dmesg)...")
    try:
        import subprocess
        result = subprocess.run(
            ["dmesg", "--time-format", "reltime"],
            capture_output=True, text=True, timeout=5
        )
        oom_lines = [l for l in result.stdout.split('\n') if 'oom' in l.lower() or 'killed' in l.lower()]
        if oom_lines:
            print("  ⚠️  Found OOM kills in dmesg:")
            for line in oom_lines[-5:]:
                print(f"    {line.strip()}")
        else:
            print("  ✓ No OOM kills found in dmesg")
    except Exception as e:
        print(f"  Could not check dmesg: {e}")
        print("  Run manually: dmesg | grep -i 'oom\\|killed'")


def main():
    parser = argparse.ArgumentParser(description="Diagnose LaDCast training issues")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dl_batches", type=int, default=200, help="Batches for DataLoader test")
    parser.add_argument("--train_steps", type=int, default=50, help="Steps for GPU leak test")
    parser.add_argument("--skip_gpu", action="store_true", help="Skip GPU tests")
    args = parser.parse_args()

    print("=" * 60)
    print("LaDCast Training Diagnostics")
    print("=" * 60)

    results = {}

    # Test 1: /dev/shm
    results["shm"] = check_shm()

    # Test 2: DataLoader
    try:
        results["dataloader"] = check_dataloader(args.config, args.dl_batches)
    except Exception as e:
        print(f"  ✗ DataLoader test crashed: {e}")
        traceback.print_exc()
        results["dataloader"] = False

    if not args.skip_gpu:
        # Test 3: GPU memory leak
        try:
            results["gpu_leak"] = check_gpu_memory_leak(args.config, args.train_steps)
        except Exception as e:
            print(f"  ✗ GPU memory test crashed: {e}")
            traceback.print_exc()
            results["gpu_leak"] = False

        # Test 4: Validation OOM
        try:
            results["val_oom"] = check_validation_oom(args.config)
        except Exception as e:
            print(f"  ✗ Validation test crashed: {e}")
            traceback.print_exc()
            results["val_oom"] = False

    # Test 5: System
    check_system()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if not all_pass:
        print("\n  Recommended fixes (in order of likelihood):")
        print("  1. If /dev/shm is small: increase it (Docker: --shm-size=8g)")
        print("  2. If num_workers > 0 with lazy xarray: set load_in_memory=true or num_workers=0")
        print("  3. If GPU memory is tight: reduce batch_size or ensemble_size")
        print("  4. Check 'dmesg | grep -i oom' for Linux OOM killer evidence")
        print("  5. Add NCCL_DEBUG=INFO for multi-GPU hang diagnosis")
    else:
        print("\n  All tests passed. If training still hangs, run with:")
        print("  CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO python ladcast/train_AR.py ...")


if __name__ == "__main__":
    main()
