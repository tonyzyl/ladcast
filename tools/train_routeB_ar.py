import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_latent_dataset import RouteBLatentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RouteB latent AR training (single process, minimal).")
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--latent_norm_json", type=str, required=True)

    parser.add_argument("--train_start_time", type=str, required=True)
    parser.add_argument("--train_end_time", type=str, required=True)
    parser.add_argument("--valid_start_time", type=str, required=True)
    parser.add_argument("--valid_end_time", type=str, required=True)

    parser.add_argument("--input_seq_len", type=int, default=1)
    parser.add_argument("--return_seq_len", type=int, default=1)
    parser.add_argument("--interval_between_pred", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=20000)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="~/ladcast/checkpoints/routeB_ar")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
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


class TinyLatentAR(nn.Module):
    """Minimal latent AR predictor: (B,Sin,C,H,W) -> (B,Sout,C,H,W)."""

    def __init__(self, in_seq: int, out_seq: int, channels: int):
        super().__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.channels = channels
        self.proj = nn.Conv2d(in_seq * channels, out_seq * channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = x.shape
        x = x.reshape(b, s * c, h, w)
        y = self.proj(x)
        y = y.reshape(b, self.out_seq, c, h, w)
        return y


def check_batch_shapes(batch: dict, input_seq_len: int, return_seq_len: int) -> None:
    x_in = batch["x_in"]
    x_out = batch["x_out"]
    if x_in.ndim != 5 or x_out.ndim != 5:
        raise RuntimeError(
            f"Expected x_in/x_out to be 5D, got {x_in.ndim} and {x_out.ndim}."
        )
    if x_in.shape[1] != input_seq_len:
        raise RuntimeError(
            f"x_in seq mismatch: expected {input_seq_len}, got {x_in.shape[1]}"
        )
    if x_out.shape[1] != return_seq_len:
        raise RuntimeError(
            f"x_out seq mismatch: expected {return_seq_len}, got {x_out.shape[1]}"
        )
    if x_in.shape[2:] != x_out.shape[2:]:
        raise RuntimeError(
            f"Spatial/channel mismatch between x_in {tuple(x_in.shape)} and x_out {tuple(x_out.shape)}"
        )


@torch.no_grad()
def validate(
        model: nn.Module,
        val_loader: DataLoader,
        val_batches: int,
        loss_fn: nn.Module,
        device: torch.device,
        input_seq_len: int,
        return_seq_len: int,
) -> float:
    model.eval()
    losses = []

    val_iter = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)

        check_batch_shapes(batch, input_seq_len, return_seq_len)

        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)

        if not torch.isfinite(x_in).all() or not torch.isfinite(x_out).all():
            raise RuntimeError("Non-finite latent tensors in validation batch.")

        pred = model(x_in)
        loss = loss_fn(pred, x_out)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss: {loss.item()}")

        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


def save_checkpoint(
        ckpt_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        train_loss: float,
        valid_loss: float,
        args: argparse.Namespace,
        channels: int,
        spatial: tuple[int, int],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "config": {
                "latent_path": args.latent_path,
                "latent_norm_json": args.latent_norm_json,
                "train_start_time": args.train_start_time,
                "train_end_time": args.train_end_time,
                "valid_start_time": args.valid_start_time,
                "valid_end_time": args.valid_end_time,
                "input_seq_len": args.input_seq_len,
                "return_seq_len": args.return_seq_len,
                "interval_between_pred": args.interval_between_pred,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
            "channels": channels,
            "spatial": spatial,
        },
        ckpt_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.log_every <= 0 or args.val_every <= 0 or args.save_every <= 0:
        raise ValueError("log_every, val_every, save_every must be > 0")
    if args.val_batches <= 0:
        raise ValueError("val_batches must be > 0")

    device = torch.device(args.device)
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = RouteBLatentDataset(
        latent_path=args.latent_path,
        start_time=args.train_start_time,
        end_time=args.train_end_time,
        input_seq_len=args.input_seq_len,
        return_seq_len=args.return_seq_len,
        interval_between_pred=args.interval_between_pred,
        normalize=True,
        latent_norm_json=args.latent_norm_json,
        return_time=False,
    )
    valid_dataset = RouteBLatentDataset(
        latent_path=args.latent_path,
        start_time=args.valid_start_time,
        end_time=args.valid_end_time,
        input_seq_len=args.input_seq_len,
        return_seq_len=args.return_seq_len,
        interval_between_pred=args.interval_between_pred,
        normalize=True,
        latent_norm_json=args.latent_norm_json,
        return_time=False,
    )

    if len(train_dataset) <= 0:
        raise RuntimeError("Train dataset length is zero.")
    if len(valid_dataset) <= 0:
        raise RuntimeError("Valid dataset length is zero.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        multiprocessing_context='spawn',
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        multiprocessing_context='spawn',
    )

    sample0 = train_dataset[0]
    check_batch_shapes(
        {"x_in": sample0["x_in"].unsqueeze(0), "x_out": sample0["x_out"].unsqueeze(0)},
        args.input_seq_len,
        args.return_seq_len,
    )
    channels = int(sample0["x_in"].shape[1])
    spatial = (int(sample0["x_in"].shape[2]), int(sample0["x_in"].shape[3]))

    print(
        f"train_len={len(train_dataset)} valid_len={len(valid_dataset)} "
        f"channels={channels} spatial={spatial}"
    )

    model = TinyLatentAR(args.input_seq_len, args.return_seq_len, channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    start_step = 1
    best_valid_loss = float("inf")
    latest_valid_loss = float("nan")

    if args.resume_from is not None:
        resume_path = os.path.expanduser(args.resume_from)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        last_step = int(ckpt.get("step", 0))
        start_step = last_step + 1
        best_valid_loss = float(ckpt.get("valid_loss", float("inf")))
        latest_valid_loss = float(ckpt.get("valid_loss", float("nan")))
        print(f"Resumed from {resume_path}, last_step={last_step}")

    model.train()
    train_iter = infinite_loader(train_loader)

    train_loss = float("nan")

    for step in range(start_step, args.max_steps + 1):
        batch = next(train_iter)
        check_batch_shapes(batch, args.input_seq_len, args.return_seq_len)

        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)

        if not torch.isfinite(x_in).all() or not torch.isfinite(x_out).all():
            raise RuntimeError(f"Non-finite latent tensors in train batch at step {step}")

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_in)
        loss = loss_fn(pred, x_out)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())

        if step % args.log_every == 0 or step == 1:
            print(f"step={step:06d} train_loss={train_loss:.8f}")

        if step % args.val_every == 0:
            latest_valid_loss = validate(
                model,
                valid_loader,
                args.val_batches,
                loss_fn,
                device,
                args.input_seq_len,
                args.return_seq_len,
            )
            print(f"step={step:06d} valid_loss={latest_valid_loss:.8f}")

            if latest_valid_loss < best_valid_loss:
                best_valid_loss = latest_valid_loss
                best_path = os.path.join(checkpoint_dir, "routeB_ar_best.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    step,
                    train_loss,
                    latest_valid_loss,
                    args,
                    channels,
                    spatial,
                )
                print(f"saved_best={best_path}")

        if step % args.save_every == 0:
            step_path = os.path.join(checkpoint_dir, f"routeB_ar_step_{step:06d}.pt")
            latest_path = os.path.join(checkpoint_dir, "routeB_ar_latest.pt")
            save_checkpoint(
                step_path,
                model,
                optimizer,
                step,
                train_loss,
                latest_valid_loss,
                args,
                channels,
                spatial,
            )
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                step,
                train_loss,
                latest_valid_loss,
                args,
                channels,
                spatial,
            )
            print(f"saved_step={step_path}")
            print(f"saved_latest={latest_path}")

    final_step = args.max_steps if args.max_steps >= start_step else start_step - 1
    final_path = os.path.join(checkpoint_dir, "routeB_ar_final.pt")
    save_checkpoint(
        final_path,
        model,
        optimizer,
        final_step,
        train_loss,
        latest_valid_loss,
        args,
        channels,
        spatial,
    )
    print(f"saved_final={final_path}")


if __name__ == "__main__":
    main()