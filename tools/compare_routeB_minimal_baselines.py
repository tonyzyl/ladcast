import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_latent_dataset import RouteBLatentDataset
from ladcast.models.routeB_latent_ar import RouteBNonSymmResNet, TinyLatentAR
from ladcast.models.routeB_symdiff_denoiser import RouteBSymDiffDenoiser
from ladcast.pipelines.pipeline_routeB_symdiff import RouteBSymDiffPipeline
from tools.train_routeB_symdiff import sample_training_group_state


@dataclass
class AblationMetric:
    mode: str
    valid_latent_mse: float


@dataclass
class BaselineMetric:
    model_name: str
    train_metric_last: float
    valid_latent_mse: float
    seconds_total: float
    seconds_per_step: float
    param_count: int
    valid_crps: float = 0.0
    spread_var: float = 0.0
    skill_mse: float = 0.0
    spread_skill_ratio: float = 0.0
    inference_ablation: list[AblationMetric] = field(default_factory=list)


@dataclass
class ComparisonReport:
    settings: dict
    baselines: list[BaselineMetric]
    result_table: str
    conclusion_a: str
    conclusion_b: str
    conclusion_c: str
    conclusion_d: str
    conclusion_e: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare minimal RouteB latent baselines under matched settings.")
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
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--time_embed_dim", type=int, default=256)
    parser.add_argument("--max_lon_shift", type=int, default=16)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--crps_samples", type=int, default=8, help="Ensemble members for CRPS/spread-skill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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


def create_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataset(args: argparse.Namespace, split: str) -> RouteBLatentDataset:
    start_time = args.train_start_time if split == "train" else args.valid_start_time
    end_time = args.train_end_time if split == "train" else args.valid_end_time
    return RouteBLatentDataset(
        latent_path=args.latent_path,
        start_time=start_time,
        end_time=end_time,
        input_seq_len=args.input_seq_len,
        return_seq_len=args.return_seq_len,
        interval_between_pred=args.interval_between_pred,
        normalize=True,
        latent_norm_json=args.latent_norm_json,
        return_time=False,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_ar(model: nn.Module, val_loader: DataLoader, val_batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    iterator = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        pred = model(x_in)
        losses.append(float(F.mse_loss(pred.float(), x_out.float()).item()))
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def validate_diffusion(
        model: RouteBSymDiffDenoiser,
        scheduler: DDPMScheduler,
        val_loader: DataLoader,
        val_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
        inference_mode: str,
        fixed_shift: int = 0,
        num_symmetry_samples: int = 1,
) -> float:
    model.eval()
    pipeline = RouteBSymDiffPipeline(denoiser=model, scheduler=scheduler)
    losses = []
    iterator = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
        cond = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        pred = pipeline(
            cond,
            num_inference_steps=num_inference_steps,
            max_lon_shift=max_lon_shift,
            inference_mode=inference_mode,
            fixed_shift=fixed_shift,
            num_symmetry_samples=num_symmetry_samples,
        )
        losses.append(float(F.mse_loss(pred.float(), x_out.float()).item()))
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# CRPS + Spread-Skill (Task 5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_crps_and_spread_skill_ar(
        model: nn.Module, val_loader: DataLoader, val_batches: int, device: torch.device,
) -> tuple[float, float, float, float]:
    """For deterministic AR: CRPS = MAE, spread = 0."""
    model.eval()
    mae_list = []
    iterator = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
        pred = model(batch["x_in"].to(device))
        target = batch["x_out"].to(device)
        mae_list.append(float((pred.float() - target.float()).abs().mean().item()))
    model.train()
    crps = float(np.mean(mae_list))
    return crps, 0.0, crps ** 2, 0.0  # crps, spread, skill, ratio


@torch.no_grad()
def compute_crps_and_spread_skill_diffusion(
        model: RouteBSymDiffDenoiser,
        scheduler: DDPMScheduler,
        val_loader: DataLoader,
        val_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
        inference_mode: str,
        n_samples: int,
) -> tuple[float, float, float, float]:
    """Compute CRPS, spread variance, skill MSE, spread-skill ratio."""
    model.eval()
    pipeline = RouteBSymDiffPipeline(denoiser=model, scheduler=scheduler)

    all_members = []  # list of (N, S, C, H, W) per member
    all_targets = []

    iterator = iter(val_loader)
    for _ in range(val_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
        cond = batch["x_in"].to(device)
        target = batch["x_out"].to(device)
        all_targets.append(target.cpu())

        members_batch = []
        for _ in range(n_samples):
            pred = pipeline.sample_once(
                cond, num_inference_steps=num_inference_steps,
                max_lon_shift=max_lon_shift, inference_mode=inference_mode,
            )
            members_batch.append(pred.cpu())
        all_members.append(torch.stack(members_batch, dim=0))  # (K, B, S, C, H, W)

    # Concatenate: members (K, N_total, S, C, H, W), targets (N_total, S, C, H, W)
    members = torch.cat(all_members, dim=1).float()  # (K, N, S, C, H, W)
    targets = torch.cat(all_targets, dim=0).float()

    K = members.shape[0]

    # CRPS = E|X-Y| - 0.5 * E|X-X'|
    term1 = float((members - targets.unsqueeze(0)).abs().mean().item())
    pairwise_sum = 0.0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            pairwise_sum += float((members[i] - members[j]).abs().mean().item())
            n_pairs += 1
    term2 = pairwise_sum / max(n_pairs, 1)
    crps = term1 - 0.5 * term2

    # Spread & Skill
    ens_mean = members.mean(dim=0)
    skill_mse = float(F.mse_loss(ens_mean, targets).item())
    spread_var = float(members.var(dim=0).mean().item())
    ratio = spread_var ** 0.5 / max(skill_mse ** 0.5, 1e-12)

    model.train()
    return crps, spread_var, skill_mse, ratio


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------

def run_tiny_ar(args, train_loader, valid_loader, channels, device) -> BaselineMetric:
    set_seed(args.seed)
    model = TinyLatentAR(args.input_seq_len, args.return_seq_len, channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = infinite_loader(train_loader)
    start = time.perf_counter()
    train_metric_last = float("nan")
    for _ in range(args.max_steps):
        batch = next(train_iter)
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_in)
        loss = F.mse_loss(pred.float(), x_out.float())
        loss.backward()
        optimizer.step()
        train_metric_last = float(loss.item())
    elapsed = time.perf_counter() - start
    valid_mse = validate_ar(model, valid_loader, args.val_batches, device)
    crps, spread, skill, ratio = compute_crps_and_spread_skill_ar(model, valid_loader, args.val_batches, device)
    return BaselineMetric("tiny_ar", train_metric_last, valid_mse, elapsed, elapsed / args.max_steps,
                          count_parameters(model), crps, spread, skill, ratio)


def run_non_symm_resnet(args, train_loader, valid_loader, channels, device) -> BaselineMetric:
    set_seed(args.seed)
    model = RouteBNonSymmResNet(
        args.input_seq_len, args.return_seq_len, channels,
        hidden_channels=args.hidden_channels, num_blocks=args.num_blocks,
        kernel_size=args.kernel_size, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = infinite_loader(train_loader)
    start = time.perf_counter()
    train_metric_last = float("nan")
    for _ in range(args.max_steps):
        batch = next(train_iter)
        x_in = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_in)
        loss = F.mse_loss(pred.float(), x_out.float())
        loss.backward()
        optimizer.step()
        train_metric_last = float(loss.item())
    elapsed = time.perf_counter() - start
    valid_mse = validate_ar(model, valid_loader, args.val_batches, device)
    crps, spread, skill, ratio = compute_crps_and_spread_skill_ar(model, valid_loader, args.val_batches, device)
    return BaselineMetric("non_symm_resnet", train_metric_last, valid_mse, elapsed, elapsed / args.max_steps,
                          count_parameters(model), crps, spread, skill, ratio)


def run_diffusion(
        args, train_loader, valid_loader, channels, device,
        model_name: str, symmetry_mode: str, symmetry_type: str = "roll",
) -> BaselineMetric:
    set_seed(args.seed)
    model = RouteBSymDiffDenoiser(
        channels=channels, cond_seq_len=args.input_seq_len, target_seq_len=args.return_seq_len,
        hidden_channels=args.hidden_channels, num_blocks=args.num_blocks,
        kernel_size=args.kernel_size, dropout=args.dropout, time_embed_dim=args.time_embed_dim,
        symmetry_type=symmetry_type,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon", clip_sample=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = infinite_loader(train_loader)
    start = time.perf_counter()
    train_metric_last = float("nan")
    for step_idx in range(args.max_steps):
        batch = next(train_iter)
        cond = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)

        # Augmentation: roll data before training
        if symmetry_mode == "augmentation":
            W = cond.shape[-1]
            k = torch.randint(0, W, (1,)).item()
            cond = torch.roll(cond, shifts=k, dims=-1)
            x_out = torch.roll(x_out, shifts=k, dims=-1)

        eps = torch.randn_like(x_out)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (cond.shape[0],), device=device)
        noisy = scheduler.add_noise(x_out, eps, t)

        if symmetry_mode == "stochastic":
            group_state = sample_training_group_state(cond.shape[0], device, args.max_lon_shift, "stochastic")
        elif symmetry_mode == "fourier_haar":
            group_state = torch.rand(cond.shape[0], device=device) * 2 * math.pi
        else:
            group_state = None

        optimizer.zero_grad(set_to_none=True)
        pred_eps = model(cond, noisy, t, group_state=group_state)
        loss = F.mse_loss(pred_eps.float(), eps.float())
        loss.backward()
        optimizer.step()
        train_metric_last = float(loss.item())
    elapsed = time.perf_counter() - start

    # Determine inference mode
    if symmetry_type == "fourier":
        valid_mode = "fourier_random_single"
    elif symmetry_mode == "identity" or symmetry_mode == "augmentation":
        valid_mode = "identity"
    else:
        valid_mode = "random_single"

    valid_mse = validate_diffusion(
        model, scheduler, valid_loader, args.val_batches, device,
        args.max_lon_shift, args.num_inference_steps, valid_mode,
    )

    # CRPS + Spread-Skill
    crps, spread, skill, ratio = compute_crps_and_spread_skill_diffusion(
        model, scheduler, valid_loader, args.val_batches, device,
        args.max_lon_shift, args.num_inference_steps, valid_mode, args.crps_samples,
    )

    # Inference ablation for stochastic roll mode
    inference_ablation = []
    if symmetry_mode == "stochastic":
        for mode, fixed_shift, num_samples in [
            ("identity", 0, 1), ("random_single", 0, 1), ("fixed_group", 1, 1),
            ("group_mean_4", 0, 4), ("group_mean_8", 0, 8),
        ]:
            inf_mode = mode if not mode.startswith("group_mean") else "group_mean"
            mse = validate_diffusion(
                model, scheduler, valid_loader, args.val_batches, device,
                args.max_lon_shift, args.num_inference_steps, inf_mode,
                fixed_shift=fixed_shift, num_symmetry_samples=num_samples,
            )
            inference_ablation.append(AblationMetric(mode=mode, valid_latent_mse=mse))

    # Fourier ablation
    if symmetry_mode == "fourier_haar":
        for mode, ns in [
            ("fourier_identity", 1), ("fourier_random_single", 1),
            ("fourier_mean", 4), ("fourier_mean", 8),
        ]:
            label = mode if ns == 1 else f"{mode}_{ns}"
            mse = validate_diffusion(
                model, scheduler, valid_loader, args.val_batches, device,
                args.max_lon_shift, args.num_inference_steps, mode,
                num_symmetry_samples=ns,
            )
            inference_ablation.append(AblationMetric(mode=label, valid_latent_mse=mse))

    return BaselineMetric(model_name, train_metric_last, valid_mse, elapsed, elapsed / args.max_steps,
                          count_parameters(model), crps, spread, skill, ratio, inference_ablation)


def build_markdown_table(metrics: list[BaselineMetric]) -> str:
    lines = [
        "| model | train metric | valid MSE | CRPS | spread | skill MSE | S/S ratio | sec/step | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        ss = f"{m.spread_skill_ratio:.3f}" if m.spread_skill_ratio > 0 else "N/A"
        lines.append(
            f"| {m.model_name} | {m.train_metric_last:.6f} | {m.valid_latent_mse:.6f} "
            f"| {m.valid_crps:.6f} | {m.spread_var:.6f} | {m.skill_mse:.6f} "
            f"| {ss} | {m.seconds_per_step:.4f} | {m.param_count} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.input_seq_len != 1 or args.return_seq_len != 1 or args.interval_between_pred != 1:
        raise ValueError("Minimal comparison only supports 1->1 single-step latent prediction.")

    set_seed(args.seed)
    device = torch.device(args.device)
    train_dataset = build_dataset(args, "train")
    valid_dataset = build_dataset(args, "valid")
    train_loader = create_dataloader(train_dataset, args.batch_size, True, args.num_workers, True)
    valid_loader = create_dataloader(valid_dataset, args.batch_size, False, args.num_workers, False)
    channels = int(train_dataset[0]["x_in"].shape[1])

    print("Running baselines...")
    metrics = [
        run_tiny_ar(args, train_loader, valid_loader, channels, device),
        run_non_symm_resnet(args, train_loader, valid_loader, channels, device),
        run_diffusion(args, train_loader, valid_loader, channels, device, "routeB_diffusion_nonsymm", "identity"),
        run_diffusion(args, train_loader, valid_loader, channels, device, "routeB_diffusion_aug", "augmentation"),
        run_diffusion(args, train_loader, valid_loader, channels, device, "routeB_symdiff", "stochastic"),
        run_diffusion(args, train_loader, valid_loader, channels, device, "routeB_symdiff_fourier_haar", "fourier_haar", "fourier"),
    ]
    metric_by_name = {m.model_name: m for m in metrics}

    best_ar = min(metric_by_name["tiny_ar"].valid_latent_mse, metric_by_name["non_symm_resnet"].valid_latent_mse)
    diff_nonsymm = metric_by_name["routeB_diffusion_nonsymm"].valid_latent_mse
    diff_symm = metric_by_name["routeB_symdiff"].valid_latent_mse
    diff_aug = metric_by_name["routeB_diffusion_aug"].valid_latent_mse
    fourier_haar = metric_by_name["routeB_symdiff_fourier_haar"].valid_latent_mse

    # Conclusion A: diffusion vs AR
    conclusion_a = (
        f"Diffusion beats the best supervised predictor by {best_ar - min(diff_nonsymm, diff_symm):.6f} latent MSE"
        if min(diff_nonsymm, diff_symm) < best_ar
        else f"Diffusion does not beat the best supervised predictor; gap={min(diff_nonsymm, diff_symm) - best_ar:.6f} latent MSE"
    )
    # Conclusion B: SymDiff roll vs nonsymm
    conclusion_b = (
        f"Symmetry-aware diffusion improves over non-symmetric diffusion by {diff_nonsymm - diff_symm:.6f} latent MSE"
        if diff_symm < diff_nonsymm
        else f"Symmetry-aware diffusion does not improve over non-symmetric diffusion; gap={diff_symm - diff_nonsymm:.6f} latent MSE"
    )
    # Conclusion C: Fourier vs Roll SymDiff
    conclusion_c = (
        f"Fourier SymDiff IMPROVES over roll SymDiff by {diff_symm - fourier_haar:.6f} latent MSE"
        if fourier_haar < diff_symm
        else f"Fourier SymDiff does not improve over roll SymDiff; gap={fourier_haar - diff_symm:.6f} latent MSE"
    )
    # Conclusion D: augmentation explains how much of SymDiff gain
    aug_gain = diff_nonsymm - diff_aug
    symm_gain = diff_nonsymm - min(diff_symm, fourier_haar)
    if aug_gain > 0 and symm_gain > 0:
        frac = aug_gain / symm_gain
        conclusion_d = f"Data augmentation explains {frac * 100:.1f}% of SymDiff gain (aug_gain={aug_gain:.6f}, symm_gain={symm_gain:.6f})"
    else:
        conclusion_d = f"No clear gain decomposition (aug_gain={aug_gain:.6f}, symm_gain={symm_gain:.6f})"
    # Conclusion E: best CRPS mode
    diffusion_metrics = [m for m in metrics if "diffusion" in m.model_name or "symdiff" in m.model_name]
    best_crps_model = min(diffusion_metrics, key=lambda m: m.valid_crps)
    conclusion_e = f"Best CRPS model: {best_crps_model.model_name} (CRPS={best_crps_model.valid_crps:.6f})"

    result_table = build_markdown_table(metrics)
    print(result_table)
    print()
    print("A:", conclusion_a)
    print("B:", conclusion_b)
    print("C:", conclusion_c)
    print("D:", conclusion_d)
    print("E:", conclusion_e)

    report = ComparisonReport(
        settings={
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
            "max_steps": args.max_steps,
            "num_inference_steps": args.num_inference_steps,
            "crps_samples": args.crps_samples,
            "seed": args.seed,
        },
        baselines=metrics,
        result_table=result_table,
        conclusion_a=conclusion_a,
        conclusion_b=conclusion_b,
        conclusion_c=conclusion_c,
        conclusion_d=conclusion_d,
        conclusion_e=conclusion_e,
    )

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
