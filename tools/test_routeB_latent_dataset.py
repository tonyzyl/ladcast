import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_latent_dataset import RouteBLatentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RouteBLatentDataset.")
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--input_seq_len", type=int, default=1)
    parser.add_argument("--return_seq_len", type=int, default=1)
    parser.add_argument("--interval_between_pred", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--latent_norm_json", type=str, default=None)
    parser.add_argument("--num_samples_to_print", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = RouteBLatentDataset(
        latent_path=args.latent_path,
        start_time=args.start_time,
        end_time=args.end_time,
        input_seq_len=args.input_seq_len,
        return_seq_len=args.return_seq_len,
        interval_between_pred=args.interval_between_pred,
        normalize=args.normalize,
        latent_norm_json=args.latent_norm_json,
        return_time=True,
    )

    print("===== RouteBLatentDataset Test =====")
    print(f"length: {len(ds)}")

    sample = ds[0]
    print(f"sample[0] x_in shape: {tuple(sample['x_in'].shape)}")
    print(f"sample[0] x_out shape: {tuple(sample['x_out'].shape)}")

    assert torch.isfinite(sample["x_in"]).all(), "x_in contains NaN/Inf"
    assert torch.isfinite(sample["x_out"]).all(), "x_out contains NaN/Inf"

    n = min(len(ds), args.num_samples_to_print)
    print("\nFirst sample timestamps:")
    for i in range(n):
        s = ds[i]
        assert torch.isfinite(s["x_in"]).all(), f"x_in contains NaN/Inf at sample {i}"
        assert torch.isfinite(s["x_out"]).all(), f"x_out contains NaN/Inf at sample {i}"
        print(f"- idx={i}: time_in={s['time_in']} time_out={s['time_out']}")

    print("\nDataset test passed.")


if __name__ == "__main__":
    main()