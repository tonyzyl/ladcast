import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_dataset import ERA5RouteBDataset
from ladcast.models.DCAE import AutoencoderDC


ROUTEB_CHANNELS = 70
ORIG_HEIGHT = 121
PADDED_HEIGHT = 128

# RouteB fixed channel blocks
VARIABLE_BLOCKS: Dict[str, Tuple[int, int]] = {
    "geopotential": (0, 13),
    "specific_humidity": (13, 26),
    "temperature": (26, 39),
    "u_component_of_wind": (39, 52),
    "v_component_of_wind": (52, 65),
    "10m_u_component_of_wind": (65, 66),
    "10m_v_component_of_wind": (66, 67),
    "2m_temperature": (67, 68),
    "mean_sea_level_pressure": (68, 69),
    "total_precipitation_6hr": (69, 70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RouteB AE reconstruction metrics.")
    parser.add_argument("--ds_path", type=str, default="~/data/ERA5_ladcast_routeB_1979_2024.zarr")
    parser.add_argument(
        "--norm_path",
        type=str,
        default="~/ladcast/static/ERA5_routeB_normal_1979_2017.json",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(device: torch.device) -> AutoencoderDC:
    model = AutoencoderDC(
        in_channels=ROUTEB_CHANNELS,
        out_channels=ROUTEB_CHANNELS,
        static_channels=0,
        latent_channels=8,
        attention_head_dim=32,
        encoder_block_out_channels=(64, 128, 256),
        decoder_block_out_channels=(64, 128, 256),
        encoder_layers_per_block=(1, 1, 1),
        decoder_layers_per_block=(1, 1, 1),
        encoder_qkv_multiscales=((), (), ()),
        decoder_qkv_multiscales=((), (), ()),
    ).to(device)
    return model


def preprocess_input(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    assert x.ndim == 4, f"Expected x.ndim == 4, got {x.ndim}"
    assert x.shape[1] == ROUTEB_CHANNELS, (
        f"Expected x.shape[1] == {ROUTEB_CHANNELS}, got {x.shape[1]}"
    )
    if not torch.isfinite(x).all():
        raise RuntimeError("Found non-finite values in input.")

    orig_h, orig_w = x.shape[2], x.shape[3]
    if orig_h != ORIG_HEIGHT:
        raise RuntimeError(f"Expected input height {ORIG_HEIGHT}, got {orig_h}.")

    pad_bottom = PADDED_HEIGHT - ORIG_HEIGHT
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_bottom), mode="constant", value=0.0)
    return x_pad, orig_h, orig_w


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = ERA5RouteBDataset(
        ds_path=args.ds_path,
        norm_path=args.norm_path,
        start_time=args.start_time,
        end_time=args.end_time,
        normalize=True,
        return_time=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty for the selected time range.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(device)
    ckpt = torch.load(os.path.expanduser(args.checkpoint_path), map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing key: model_state_dict")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    total_abs_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    block_abs_sum = {name: 0.0 for name in VARIABLE_BLOCKS}
    block_sq_sum = {name: 0.0 for name in VARIABLE_BLOCKS}
    block_count = {name: 0 for name in VARIABLE_BLOCKS}

    seen_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_pad, orig_h, orig_w = preprocess_input(x)

            pred = model(x_pad, return_static=True).sample
            pred_cropped = pred[:, :, :orig_h, :orig_w]

            err = pred_cropped - x
            abs_err = torch.abs(err)
            sq_err = err * err

            total_abs_sum += float(abs_err.sum().item())
            total_sq_sum += float(sq_err.sum().item())
            total_count += int(abs_err.numel())

            for name, (s, e) in VARIABLE_BLOCKS.items():
                abs_block = abs_err[:, s:e, :, :]
                sq_block = sq_err[:, s:e, :, :]
                block_abs_sum[name] += float(abs_block.sum().item())
                block_sq_sum[name] += float(sq_block.sum().item())
                block_count[name] += int(abs_block.numel())

            seen_batches += 1
            if seen_batches >= args.max_batches:
                break

    if total_count == 0:
        raise RuntimeError("No samples evaluated; check max_batches or dataset.")

    overall_mae = total_abs_sum / total_count
    overall_mse = total_sq_sum / total_count

    print("===== RouteB AE Reconstruction Evaluation =====")
    print(f"batches_evaluated: {seen_batches}")
    print(f"samples_in_dataset_window: {len(dataset)}")
    print(f"overall_mse: {overall_mse:.8f}")
    print(f"overall_mae: {overall_mae:.8f}")
    print("\nPer-variable-block metrics:")

    for name in VARIABLE_BLOCKS:
        mse = block_sq_sum[name] / block_count[name]
        mae = block_abs_sum[name] / block_count[name]
        print(f"- {name:28s} mse={mse:.8f} mae={mae:.8f}")


if __name__ == "__main__":
    main()