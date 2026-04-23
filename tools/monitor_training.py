#!/usr/bin/env python3
"""
Lightweight memory monitor to run alongside train_AR.py.
Logs GPU/CPU/shm memory every N seconds to a file.

Usage:
    python tools/monitor_training.py --interval 30 --log memory_log.csv &
    # Then run training as normal
    accelerate launch ladcast/train_AR.py --config your_config.yaml ...

After crash, inspect memory_log.csv to see the trend before death.
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime

import psutil

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


def get_gpu_stats():
    """Get GPU memory stats for all GPUs."""
    if not HAS_CUDA:
        return {}
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = torch.cuda.get_device_properties(i).total_mem
        stats[f"gpu{i}_allocated_mb"] = allocated / 1024**2
        stats[f"gpu{i}_reserved_mb"] = reserved / 1024**2
        stats[f"gpu{i}_total_mb"] = total / 1024**2
        stats[f"gpu{i}_pct"] = 100 * allocated / total
    return stats


def get_nvidia_smi_stats():
    """Fallback: parse nvidia-smi for memory (works across processes)."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        stats = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(',')]
            idx = int(parts[0])
            stats[f"gpu{idx}_used_mb"] = float(parts[1])
            stats[f"gpu{idx}_total_mb"] = float(parts[2])
            stats[f"gpu{idx}_util_pct"] = float(parts[3])
        return stats
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="Seconds between samples")
    parser.add_argument("--log", type=str, default="memory_log.csv", help="Output CSV path")
    args = parser.parse_args()

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Monitoring memory every {args.interval}s → {args.log}")
    print("Press Ctrl+C to stop.\n")

    header_written = False
    while running:
        row = {"timestamp": datetime.now().isoformat()}

        # CPU / RAM
        vm = psutil.virtual_memory()
        row["ram_used_mb"] = vm.used / 1024**2
        row["ram_available_mb"] = vm.available / 1024**2
        row["ram_pct"] = vm.percent

        # /dev/shm
        if os.path.exists("/dev/shm"):
            shm = psutil.disk_usage("/dev/shm")
            row["shm_used_mb"] = shm.used / 1024**2
            row["shm_pct"] = shm.percent

        # GPU (nvidia-smi, works across all processes)
        gpu_stats = get_nvidia_smi_stats()
        row.update(gpu_stats)

        # Write CSV
        with open(args.log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)

        # Print summary
        gpu_summary = ""
        for k, v in sorted(gpu_stats.items()):
            if "used_mb" in k:
                gpu_summary += f"  {k}={v:.0f}MB"
        print(f"[{row['timestamp'][:19]}] RAM: {vm.percent}%{gpu_summary}")

        time.sleep(args.interval)

    print(f"\nStopped. Log saved to {args.log}")


if __name__ == "__main__":
    main()
