import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
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
from ladcast.models.routeB_latent_ar import RouteBNonSymmResNet, RouteBSymmResNet, TinyLatentAR


@dataclass
class InferenceMetrics:
    mode: str
    valid_loss: float


@dataclass
class ModelMetrics:
    model_type: str
    param_count: int
    train_loss_last: float
    train_loss_mean: float
    valid_loss: float
    total_train_seconds: float
    seconds_per_step: float
    stable: bool
    overfit_flag: bool
    inference_ablation: list[InferenceMetrics] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare routeB AR models under matched settings.")
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
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_lon_shift", type=int, default=16)
    parser.add_argument("--tta_samples_4", type=int, default=4)
    parser.add_argument("--tta_samples_8", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate(
        model: nn.Module,
        val_loader: DataLoader,
        val_batches: int,
        device: torch.device,
        *,
        inference_mode: str | None = None,
        num_symmetry_samples: int = 1,
) -> float:
    model.eval()
    losses = []
    loss_fn = nn.MSELoss()
    iterator = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        if inference_mode is None:
            pred = model(x_in)
        else:
            pred = model(x_in, inference_mode=inference_mode, num_symmetry_samples=num_symmetry_samples)
        loss = loss_fn(pred, x_out)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss for {model.__class__.__name__}: {loss.item()}")
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def build_model(args: argparse.Namespace, model_type: str, channels: int) -> nn.Module:
    if model_type == "tiny_ar":
        return TinyLatentAR(args.input_seq_len, args.return_seq_len, channels)
    if model_type == "non_symm_resnet":
        return RouteBNonSymmResNet(
            args.input_seq_len,
            args.return_seq_len,
            channels,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if model_type == "symm_resnet":
        return RouteBSymmResNet(
            args.input_seq_len,
            args.return_seq_len,
            channels,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            max_lon_shift=args.max_lon_shift,
        )
    raise ValueError(model_type)


def collect_inference_ablation(
        model: nn.Module,
        args: argparse.Namespace,
        valid_loader: DataLoader,
        device: torch.device,
) -> list[InferenceMetrics]:
    if not isinstance(model, RouteBSymmResNet):
        return []

    configs = [
        ("deterministic", "deterministic", 1),
        ("random_single", "random_single", 1),
        (f"group_mean_{args.tta_samples_4}", "group_mean", args.tta_samples_4),
        (f"group_mean_{args.tta_samples_8}", "group_mean", args.tta_samples_8),
    ]
    out = []
    for name, mode, n_samples in configs:
        valid_loss = validate(
            model,
            valid_loader,
            args.val_batches,
            device,
            inference_mode=mode,
            num_symmetry_samples=n_samples,
        )
        out.append(InferenceMetrics(mode=name, valid_loss=valid_loss))
    return out


def run_one_model(
        args: argparse.Namespace,
        model_type: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        channels: int,
        device: torch.device,
) -> ModelMetrics:
    set_seed(args.seed)
    model = build_model(args, model_type, channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    train_iter = infinite_loader(train_loader)
    losses = []
    stable = True

    start = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        batch = next(train_iter)
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_in)
        loss = loss_fn(pred, x_out)
        if not torch.isfinite(loss):
            stable = False
            raise RuntimeError(f"Non-finite train loss for {model_type} at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    total_train_seconds = time.perf_counter() - start

    valid_loss = validate(model, valid_loader, args.val_batches, device)
    train_loss_last = losses[-1]
    train_loss_mean = float(np.mean(losses))
    overfit_flag = valid_loss > train_loss_last * 1.20
    inference_ablation = collect_inference_ablation(model, args, valid_loader, device)
    return ModelMetrics(
        model_type=model_type,
        param_count=count_parameters(model),
        train_loss_last=train_loss_last,
        train_loss_mean=train_loss_mean,
        valid_loss=valid_loss,
        total_train_seconds=total_train_seconds,
        seconds_per_step=total_train_seconds / args.max_steps,
        stable=stable,
        overfit_flag=overfit_flag,
        inference_ablation=inference_ablation,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        multiprocessing_context="spawn",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        multiprocessing_context="spawn",
    )

    sample0 = train_dataset[0]
    channels = int(sample0["x_in"].shape[1])

    results = []
    for model_type in ["tiny_ar", "non_symm_resnet", "symm_resnet"]:
        print(f"\n=== Running {model_type} ===")
        metrics = run_one_model(args, model_type, train_loader, valid_loader, channels, device)
        results.append(metrics)
        print(json.dumps(asdict(metrics), indent=2))

    if args.output_json is not None:
        output_path = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in results], f, indent=2)
        print(f"Saved comparison report: {output_path}")


if __name__ == "__main__":
    main()
