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
    parser = argparse.ArgumentParser(description="Minimal single-process RouteB latent AR smoke train.")
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--input_seq_len", type=int, default=1)
    parser.add_argument("--return_seq_len", type=int, default=1)
    parser.add_argument("--interval_between_pred", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--latent_norm_json", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="~/ladcast/checkpoints/routeB_ar_smoke.pt",
    )
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
    """Maps input latent sequence -> predicted latent sequence using a single conv."""

    def __init__(self, in_seq: int, out_seq: int, channels: int):
        super().__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.channels = channels
        self.proj = nn.Conv2d(in_seq * channels, out_seq * channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S_in, C, H, W)
        b, s, c, h, w = x.shape
        x = x.reshape(b, s * c, h, w)
        y = self.proj(x)
        y = y.reshape(b, self.out_seq, c, h, w)
        return y


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    dataset = RouteBLatentDataset(
        latent_path=args.latent_path,
        start_time=args.start_time,
        end_time=args.end_time,
        input_seq_len=args.input_seq_len,
        return_seq_len=args.return_seq_len,
        interval_between_pred=args.interval_between_pred,
        normalize=args.normalize,
        latent_norm_json=args.latent_norm_json,
        return_time=False,
    )

    if len(dataset) <= 0:
        raise RuntimeError("RouteBLatentDataset has zero length for selected range.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    sample0 = dataset[0]
    c = int(sample0["x_in"].shape[1])
    h = int(sample0["x_in"].shape[2])
    w = int(sample0["x_in"].shape[3])
    print(f"dataset_len={len(dataset)} channels={c} spatial=({h},{w})")

    model = TinyLatentAR(args.input_seq_len, args.return_seq_len, c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    stream = infinite_loader(loader)
    model.train()

    for step in range(1, args.max_steps + 1):
        batch = next(stream)
        x_in = batch["x_in"].to(device)   # (B, Sin, C, H, W)
        x_out = batch["x_out"].to(device)  # (B, Sout, C, H, W)

        assert x_in.ndim == 5 and x_out.ndim == 5, "Expected latent sequence tensors to be 5D"
        if not torch.isfinite(x_in).all() or not torch.isfinite(x_out).all():
            raise RuntimeError(f"Non-finite input/output latent tensor at step {step}")

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_in)
        loss = loss_fn(pred, x_out)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()

        print(f"step={step:04d} loss={loss.item():.6f}")

    checkpoint_path = os.path.expanduser(args.checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_seq_len": args.input_seq_len,
            "return_seq_len": args.return_seq_len,
            "channels": c,
            "max_steps": args.max_steps,
            "final_loss": float(loss.item()),
        },
        checkpoint_path,
    )
    print(f"Saved AR smoke checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()