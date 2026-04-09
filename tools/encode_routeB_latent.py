import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_dataset import ERA5RouteBDataset
from ladcast.models.DCAE import AutoencoderDC


ROUTEB_CHANNELS = 70
ORIG_HEIGHT = 121
PADDED_HEIGHT = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode RouteB inputs into AE latent zarr.")
    parser.add_argument("--ds_path", type=str, default="~/data/ERA5_ladcast_routeB_1979_2024.zarr")
    parser.add_argument(
        "--norm_path",
        type=str,
        default="~/ladcast/static/ERA5_routeB_normal_1979_2017.json",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--output_zarr", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
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


def preprocess_input(x: torch.Tensor):
    assert x.ndim == 4, f"Expected x.ndim == 4, got {x.ndim}"
    assert x.shape[1] == ROUTEB_CHANNELS, (
        f"Expected x.shape[1] == {ROUTEB_CHANNELS}, got {x.shape[1]}"
    )
    if not torch.isfinite(x).all():
        raise RuntimeError("Found non-finite values in input.")

    orig_h = x.shape[2]
    if orig_h != ORIG_HEIGHT:
        raise RuntimeError(f"Expected input height {ORIG_HEIGHT}, got {orig_h}.")

    pad_bottom = PADDED_HEIGHT - ORIG_HEIGHT
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_bottom), mode="constant", value=0.0)
    return x_pad


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = ERA5RouteBDataset(
        ds_path=args.ds_path,
        norm_path=args.norm_path,
        start_time=args.start_time,
        end_time=args.end_time,
        normalize=True,
        return_time=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty for the selected time range.")

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

    latents_all = []
    times_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_pad = preprocess_input(x)

            z = model.encode(x_pad, return_dict=False)[0]  # (B, latent_channels, H_lat, W_lat)
            z_np = z.detach().cpu().numpy().astype(np.float32)
            latents_all.append(z_np)

            batch_times = batch["time"]
            times_all.extend([str(t) for t in batch_times])

    latent_array = np.concatenate(latents_all, axis=0)

    num_samples, latent_channels, h_lat, w_lat = latent_array.shape
    if num_samples != len(times_all):
        raise RuntimeError("Mismatch between latent samples and collected timestamps.")

    ds = xr.Dataset(
        data_vars={
            "latent": (
                ("time", "channel", "lat", "lon"),
                latent_array,
            )
        },
        coords={
            "time": np.array(times_all, dtype="datetime64[ns]"),
            "channel": np.arange(latent_channels, dtype=np.int32),
            "lat": np.arange(h_lat, dtype=np.int32),
            "lon": np.arange(w_lat, dtype=np.int32),
        },
        attrs={
            "source": "routeB AE encode",
            "input_channels": ROUTEB_CHANNELS,
            "input_height": ORIG_HEIGHT,
            "padded_height": PADDED_HEIGHT,
        },
    )

    output_path = os.path.expanduser(args.output_zarr)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_zarr(output_path, mode="w")

    print("===== RouteB latent encoding done =====")
    print(f"num_samples: {num_samples}")
    print(f"latent_shape: {latent_array.shape}")
    print(f"output_zarr: {output_path}")


if __name__ == "__main__":
    main()