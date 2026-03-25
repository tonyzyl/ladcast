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
    parser = argparse.ArgumentParser(
        description="Encode RouteB inputs into latent zarr (large-range, batch append write)."
    )
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--norm_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing zarr instead of failing when out_path exists.",
    )
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


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
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


def batch_to_dataset(z: np.ndarray, times: list[str]) -> xr.Dataset:
    if not np.isfinite(z).all():
        raise RuntimeError("Encoded latent contains NaN/Inf.")
    t = np.array(times, dtype="datetime64[ns]")
    b, c, h, w = z.shape
    return xr.Dataset(
        data_vars={
            "latent": (("time", "channel", "lat", "lon"), z),
        },
        coords={
            "time": t,
            "channel": np.arange(c, dtype=np.int32),
            "lat": np.arange(h, dtype=np.int32),
            "lon": np.arange(w, dtype=np.int32),
        },
        attrs={
            "source": "routeB AE encode large",
            "input_channels": ROUTEB_CHANNELS,
            "input_height": ORIG_HEIGHT,
            "padded_height": PADDED_HEIGHT,
        },
    )


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

    out_path = os.path.expanduser(args.out_path)
    out_exists = os.path.exists(out_path)
    if out_exists and not args.append:
        raise FileExistsError(
            f"Output zarr already exists: {out_path}. Use --append to append."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total_samples = 0
    first_write = not out_exists

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device)
            x_pad = preprocess_input(x)
            z = model.encode(x_pad, return_dict=False)[0]  # (B, C, H, W)
            z_np = z.detach().cpu().numpy().astype(np.float32)
            batch_times = [str(t) for t in batch["time"]]

            batch_ds = batch_to_dataset(z_np, batch_times)

            if first_write:
                batch_ds.to_zarr(out_path, mode="w")
                first_write = False
            else:
                batch_ds.to_zarr(out_path, mode="a", append_dim="time")

            total_samples += z_np.shape[0]
            print(
                f"batch={batch_idx:06d} encoded={z_np.shape[0]} "
                f"latent_shape={z_np.shape} total={total_samples}"
            )

    print("===== RouteB large latent encoding done =====")
    print(f"samples_encoded: {total_samples}")
    print(f"output_zarr: {out_path}")


if __name__ == "__main__":
    main()