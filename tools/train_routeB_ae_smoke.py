import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from torch import nn
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
    parser = argparse.ArgumentParser(description="RouteB AE training (single process, minimal).")
    parser.add_argument("--ds_path", type=str, default="~/data/ERA5_ladcast_routeB_1979_2024.zarr")
    parser.add_argument(
        "--norm_path",
        type=str,
        default="~/ladcast/static/ERA5_routeB_normal_1979_2017.json",
    )

    parser.add_argument("--train_start_time", type=str, required=True)
    parser.add_argument("--train_end_time", type=str, required=True)
    parser.add_argument("--valid_start_time", type=str, required=True)
    parser.add_argument("--valid_end_time", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=10)

    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="~/ladcast/checkpoints/routeB_ae")
    parser.add_argument("--save_latest_name", type=str, default="routeB_ae_latest.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite_loader(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


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
    if pad_bottom < 0:
        raise RuntimeError(
            f"PADDED_HEIGHT ({PADDED_HEIGHT}) must be >= ORIG_HEIGHT ({ORIG_HEIGHT})."
        )

    x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_bottom), mode="constant", value=0.0)
    return x_pad, orig_h, orig_w


def train_step(model, batch, optimizer, loss_fn, device) -> float:
    x = batch["x"].to(device)
    x_pad, orig_h, orig_w = preprocess_input(x)

    optimizer.zero_grad(set_to_none=True)
    pred = model(x_pad, return_static=True).sample
    pred_cropped = pred[:, :, :orig_h, :orig_w]

    loss = loss_fn(pred_cropped, x)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Train loss is non-finite: {loss.item()}")

    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def validate(model, val_loader, val_batches, loss_fn, device) -> float:
    model.eval()
    losses = []

    it = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_loader)
            batch = next(it)

        x = batch["x"].to(device)
        x_pad, orig_h, orig_w = preprocess_input(x)
        pred = model(x_pad, return_static=True).sample
        pred_cropped = pred[:, :, :orig_h, :orig_w]
        loss = loss_fn(pred_cropped, x)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Validation loss is non-finite: {loss.item()}")
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: str, model, optimizer, step: int, train_loss: float, val_loss: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "in_channels": ROUTEB_CHANNELS,
            "out_channels": ROUTEB_CHANNELS,
            "static_channels": 0,
            "orig_height": ORIG_HEIGHT,
            "padded_height": PADDED_HEIGHT,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    ckpt_dir = os.path.expanduser(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = ERA5RouteBDataset(
        ds_path=args.ds_path,
        norm_path=args.norm_path,
        start_time=args.train_start_time,
        end_time=args.train_end_time,
        normalize=True,
        return_time=False,
    )
    valid_dataset = ERA5RouteBDataset(
        ds_path=args.ds_path,
        norm_path=args.norm_path,
        start_time=args.valid_start_time,
        end_time=args.valid_end_time,
        normalize=True,
        return_time=False,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty for the selected time range.")
    if len(valid_dataset) == 0:
        raise RuntimeError("Valid dataset is empty for the selected time range.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    train_stream = infinite_loader(train_loader)

    best_val = float("inf")
    latest_val = float("nan")

    for step in range(1, args.max_steps + 1):
        train_loss = train_step(model, next(train_stream), optimizer, loss_fn, device)

        if step % args.log_every == 0 or step == 1:
            print(f"step={step:06d} train_loss={train_loss:.6f}")

        if step % args.val_every == 0:
            latest_val = validate(model, valid_loader, args.val_batches, loss_fn, device)
            print(f"step={step:06d} valid_loss={latest_val:.6f}")
            if latest_val < best_val:
                best_val = latest_val
                best_path = os.path.join(ckpt_dir, "routeB_ae_best.pt")
                save_checkpoint(best_path, model, optimizer, step, train_loss, latest_val)
                print(f"saved_best={best_path}")

        if step % args.save_every == 0:
            step_path = os.path.join(ckpt_dir, f"routeB_ae_step_{step:06d}.pt")
            latest_path = os.path.join(ckpt_dir, args.save_latest_name)
            save_checkpoint(step_path, model, optimizer, step, train_loss, latest_val)
            save_checkpoint(latest_path, model, optimizer, step, train_loss, latest_val)
            print(f"saved_ckpt={step_path}")
            print(f"saved_latest={latest_path}")

    final_path = os.path.join(ckpt_dir, "routeB_ae_final.pt")
    save_checkpoint(final_path, model, optimizer, args.max_steps, train_loss, latest_val)
    print(f"saved_final={final_path}")


if __name__ == "__main__":
    main()