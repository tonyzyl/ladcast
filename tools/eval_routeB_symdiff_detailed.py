"""
Detailed evaluation for RouteB SymDiff model.

Loads a trained checkpoint and produces comprehensive metrics:
  1. Overall latent MSE (+ per-channel breakdown)
  2. Spatial error maps (per lat/lon grid cell)
  3. Ensemble spread-skill analysis (CRPS-like via group_mean samples)
  4. Inference mode ablation (identity / random_single / fixed_group / group_mean_K)
  5. Multi-member ensemble statistics (spread, rank histogram proxy)
  6. Timing benchmark per inference call

Usage:
  python tools/eval_routeB_symdiff_detailed.py \
    --checkpoint ~/ladcast/checkpoints/routeB_diffusion/<run_name>_best.pt \
    --latent_path ~/ladcast/data/routeB_latent_train.zarr \
    --latent_norm_json ~/ladcast/static/ERA5_routeB_latent_normal_train.json \
    --valid_start_time 2018-02-01T00 \
    --valid_end_time 2018-02-07T23 \
    --val_batches 20 \
    --ensemble_members 8 \
    --num_inference_steps 50 \
    --output_json tmp/routeB_symdiff_detailed_eval.json \
    --device cuda
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_latent_dataset import RouteBLatentDataset
from ladcast.models.routeB_symdiff_denoiser import RouteBSymDiffDenoiser
from ladcast.pipelines.pipeline_routeB_symdiff import RouteBSymDiffPipeline


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class ChannelMetric:
    channel_idx: int
    mse: float
    rmse: float
    mae: float
    bias: float


@dataclass
class SpatialErrorMap:
    """MSE averaged over batch and channels, shape (H, W) stored as nested list."""
    shape: list[int]
    mse_map: list[list[float]]


@dataclass
class InferenceModeResult:
    mode: str
    num_symmetry_samples: int
    mse: float
    rmse: float
    mae: float
    seconds_per_batch: float


@dataclass
class EnsembleDiagnostics:
    num_members: int
    mean_mse: float
    mean_rmse: float
    spread_mean: float          # average ensemble std across grid
    spread_skill_ratio: float   # spread / RMSE (ideal ≈ 1)
    crps_approx: float          # approximate CRPS from ensemble
    member_mse_list: list[float]


@dataclass
class DetailedEvalReport:
    checkpoint: str
    model_name: str
    step: int
    num_val_samples: int
    overall_mse: float
    overall_rmse: float
    overall_mae: float
    per_channel: list[ChannelMetric]
    spatial_error: SpatialErrorMap
    inference_ablation: list[InferenceModeResult]
    ensemble_diagnostics: EnsembleDiagnostics
    config: dict


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detailed RouteB SymDiff evaluation.")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained .pt checkpoint")
    p.add_argument("--latent_path", type=str, required=True)
    p.add_argument("--latent_norm_json", type=str, required=True)
    p.add_argument("--valid_start_time", type=str, required=True)
    p.add_argument("--valid_end_time", type=str, required=True)
    p.add_argument("--input_seq_len", type=int, default=1)
    p.add_argument("--return_seq_len", type=int, default=1)
    p.add_argument("--interval_between_pred", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_batches", type=int, default=20,
                    help="Number of validation batches to evaluate")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--max_lon_shift", type=int, default=16)
    p.add_argument("--ensemble_members", type=int, default=8,
                    help="Number of ensemble members for spread/skill analysis")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", type=str, default=None,
                    help="Path to save JSON report")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs = {"batch_size": batch_size, "shuffle": False,
              "num_workers": num_workers, "drop_last": False}
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)


def iterate_batches(loader: DataLoader, n_batches: int) -> Iterator[dict]:
    """Yield exactly n_batches from loader, wrapping around if needed."""
    count = 0
    while count < n_batches:
        for batch in loader:
            yield batch
            count += 1
            if count >= n_batches:
                return


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load checkpoint, reconstruct model + scheduler, return (pipeline, ckpt_meta)."""
    ckpt_path = str(Path(ckpt_path).expanduser())
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt["config"]
    channels = ckpt["channels"]

    model = RouteBSymDiffDenoiser(
        channels=channels,
        cond_seq_len=cfg.get("input_seq_len", 1),
        target_seq_len=cfg.get("return_seq_len", 1),
        hidden_channels=cfg["hidden_channels"],
        num_blocks=cfg["num_blocks"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg.get("dropout", 0.0),
        time_embed_dim=cfg["time_embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_train_timesteps"],
        beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type="epsilon",
        clip_sample=False,
    )

    pipeline = RouteBSymDiffPipeline(denoiser=model, scheduler=scheduler)
    return pipeline, ckpt


# ---------------------------------------------------------------------------
# 1. Overall + per-channel metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_overall_and_channel_metrics(
        pipeline: RouteBSymDiffPipeline,
        loader: DataLoader,
        n_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
) -> tuple[float, float, float, list[ChannelMetric], list[torch.Tensor], list[torch.Tensor]]:
    """Returns (mse, rmse, mae, per_channel_metrics, all_preds, all_targets)."""
    all_preds = []
    all_targets = []

    for batch in iterate_batches(loader, n_batches):
        cond = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        pred = pipeline(
            cond,
            num_inference_steps=num_inference_steps,
            max_lon_shift=max_lon_shift,
            inference_mode="random_single",
        )
        all_preds.append(pred.cpu())
        all_targets.append(x_out.cpu())

    preds = torch.cat(all_preds, dim=0)    # (N, S, C, H, W)
    targets = torch.cat(all_targets, dim=0)

    diff = preds.float() - targets.float()
    overall_mse = float((diff ** 2).mean().item())
    overall_rmse = float(overall_mse ** 0.5)
    overall_mae = float(diff.abs().mean().item())

    # Per-channel: average over (N, S, H, W) keeping C
    n, s, c, h, w = diff.shape
    diff_flat = diff.reshape(-1, c, h, w)  # (N*S, C, H, W)
    channel_metrics = []
    for ci in range(c):
        ch_diff = diff_flat[:, ci, :, :]
        ch_mse = float((ch_diff ** 2).mean().item())
        ch_rmse = float(ch_mse ** 0.5)
        ch_mae = float(ch_diff.abs().mean().item())
        ch_bias = float(ch_diff.mean().item())
        channel_metrics.append(ChannelMetric(ci, ch_mse, ch_rmse, ch_mae, ch_bias))

    return overall_mse, overall_rmse, overall_mae, channel_metrics, all_preds, all_targets


# ---------------------------------------------------------------------------
# 2. Spatial error map
# ---------------------------------------------------------------------------

def compute_spatial_error_map(
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
) -> SpatialErrorMap:
    """MSE averaged over (N, S, C) dimensions → (H, W) map."""
    preds = torch.cat(all_preds, dim=0).float()
    targets = torch.cat(all_targets, dim=0).float()
    # shape: (N, S, C, H, W)
    se = (preds - targets) ** 2
    mse_map = se.mean(dim=(0, 1, 2))  # (H, W)
    h, w = mse_map.shape
    return SpatialErrorMap(
        shape=[int(h), int(w)],
        mse_map=mse_map.tolist(),
    )


# ---------------------------------------------------------------------------
# 3. Inference mode ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablation_single_mode(
        pipeline: RouteBSymDiffPipeline,
        loader: DataLoader,
        n_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
        inference_mode: str,
        fixed_shift: int = 0,
        num_symmetry_samples: int = 1,
) -> InferenceModeResult:
    mode_label = inference_mode
    if inference_mode == "group_mean":
        mode_label = f"group_mean_{num_symmetry_samples}"

    total_mse, total_mae = 0.0, 0.0
    total_time = 0.0
    count = 0

    for batch in iterate_batches(loader, n_batches):
        cond = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)

        t0 = time.perf_counter()
        pred = pipeline(
            cond,
            num_inference_steps=num_inference_steps,
            max_lon_shift=max_lon_shift,
            inference_mode=inference_mode,
            fixed_shift=fixed_shift,
            num_symmetry_samples=num_symmetry_samples,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        diff = pred.float() - x_out.float()
        total_mse += float((diff ** 2).mean().item())
        total_mae += float(diff.abs().mean().item())
        total_time += elapsed
        count += 1

    avg_mse = total_mse / max(count, 1)
    avg_mae = total_mae / max(count, 1)
    avg_rmse = avg_mse ** 0.5
    avg_time = total_time / max(count, 1)

    return InferenceModeResult(
        mode=mode_label,
        num_symmetry_samples=num_symmetry_samples,
        mse=avg_mse,
        rmse=avg_rmse,
        mae=avg_mae,
        seconds_per_batch=avg_time,
    )


def run_inference_ablation(
        pipeline: RouteBSymDiffPipeline,
        loader: DataLoader,
        n_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
) -> list[InferenceModeResult]:
    configs = [
        ("identity",      0, 1),
        ("random_single", 0, 1),
        ("fixed_group",   1, 1),
        ("fixed_group",   4, 1),
        ("fixed_group",   8, 1),
        ("group_mean",    0, 2),
        ("group_mean",    0, 4),
        ("group_mean",    0, 8),
        ("group_mean",    0, 16),
    ]
    results = []
    for mode, fs, ns in configs:
        print(f"  ablation: mode={mode}, fixed_shift={fs}, num_samples={ns} ...")
        r = ablation_single_mode(
            pipeline, loader, n_batches, device,
            max_lon_shift, num_inference_steps,
            mode, fixed_shift=fs, num_symmetry_samples=ns,
        )
        print(f"    MSE={r.mse:.6f}  RMSE={r.rmse:.6f}  MAE={r.mae:.6f}  "
              f"time={r.seconds_per_batch:.3f}s/batch")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# 4. Ensemble diagnostics (spread, skill, CRPS approximation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ensemble_diagnostics(
        pipeline: RouteBSymDiffPipeline,
        loader: DataLoader,
        n_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
        num_members: int,
) -> EnsembleDiagnostics:
    """Generate K independent ensemble members via random_single, compute
    spread/skill diagnostics and approximate CRPS."""
    all_member_preds = [[] for _ in range(num_members)]
    all_targets = []

    for batch in iterate_batches(loader, n_batches):
        cond = batch["x_in"].to(device)
        x_out = batch["x_out"].to(device)
        all_targets.append(x_out.cpu())

        for k in range(num_members):
            pred = pipeline.sample_once(
                cond,
                num_inference_steps=num_inference_steps,
                max_lon_shift=max_lon_shift,
                inference_mode="random_single",
            )
            all_member_preds[k].append(pred.cpu())

    # Stack: (K, N, S, C, H, W)
    members = torch.stack(
        [torch.cat(all_member_preds[k], dim=0) for k in range(num_members)],
        dim=0,
    ).float()
    targets = torch.cat(all_targets, dim=0).float()  # (N, S, C, H, W)

    # Ensemble mean
    ens_mean = members.mean(dim=0)  # (N, S, C, H, W)
    mean_mse = float(F.mse_loss(ens_mean, targets).item())
    mean_rmse = mean_mse ** 0.5

    # Per-member MSE
    member_mse_list = []
    for k in range(num_members):
        m_mse = float(F.mse_loss(members[k], targets).item())
        member_mse_list.append(m_mse)

    # Spread: std across members at each grid point, then average
    ens_std = members.std(dim=0)  # (N, S, C, H, W)
    spread_mean = float(ens_std.mean().item())

    # Spread-skill ratio
    spread_skill_ratio = spread_mean / max(mean_rmse, 1e-12)

    # Approximate CRPS using ensemble members (fair CRPS estimator)
    # CRPS ≈ E|X - y| - 0.5 * E|X - X'|
    # where X, X' are independent draws from the ensemble
    diff_to_obs = (members - targets.unsqueeze(0)).abs()  # (K, N, S, C, H, W)
    term1 = float(diff_to_obs.mean().item())

    # E|X - X'| estimated pairwise
    pairwise_sum = 0.0
    n_pairs = 0
    for i in range(num_members):
        for j in range(i + 1, num_members):
            pairwise_sum += float((members[i] - members[j]).abs().mean().item())
            n_pairs += 1
    term2 = pairwise_sum / max(n_pairs, 1)

    crps_approx = term1 - 0.5 * term2

    return EnsembleDiagnostics(
        num_members=num_members,
        mean_mse=mean_mse,
        mean_rmse=mean_rmse,
        spread_mean=spread_mean,
        spread_skill_ratio=spread_skill_ratio,
        crps_approx=crps_approx,
        member_mse_list=member_mse_list,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    pipeline, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    model_name = ckpt.get("model_name", "unknown")
    step = ckpt.get("step", -1)
    print(f"  model_name={model_name}, step={step}")

    print("Building validation dataset ...")
    dataset = RouteBLatentDataset(
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
    loader = create_dataloader(dataset, args.batch_size, args.num_workers)
    n_val_samples = min(args.val_batches * args.batch_size, len(dataset))
    print(f"  dataset len={len(dataset)}, eval samples≈{n_val_samples}")

    # --- 1. Overall + per-channel metrics ---
    print("\n[1/4] Computing overall & per-channel metrics ...")
    overall_mse, overall_rmse, overall_mae, ch_metrics, all_preds, all_targets = \
        compute_overall_and_channel_metrics(
            pipeline, loader, args.val_batches, device,
            args.max_lon_shift, args.num_inference_steps,
        )
    print(f"  Overall MSE={overall_mse:.6f}  RMSE={overall_rmse:.6f}  MAE={overall_mae:.6f}")
    for cm in ch_metrics:
        print(f"  ch{cm.channel_idx}: MSE={cm.mse:.6f}  RMSE={cm.rmse:.6f}  "
              f"MAE={cm.mae:.6f}  bias={cm.bias:.6f}")

    # --- 2. Spatial error map ---
    print("\n[2/4] Computing spatial error map ...")
    spatial = compute_spatial_error_map(all_preds, all_targets)
    mse_arr = np.array(spatial.mse_map)
    print(f"  Spatial MSE shape={spatial.shape}, "
          f"min={mse_arr.min():.6f}, max={mse_arr.max():.6f}, "
          f"mean={mse_arr.mean():.6f}")

    # --- 3. Inference mode ablation ---
    print("\n[3/4] Running inference mode ablation ...")
    ablation = run_inference_ablation(
        pipeline, loader, args.val_batches, device,
        args.max_lon_shift, args.num_inference_steps,
    )

    # --- 4. Ensemble diagnostics ---
    print(f"\n[4/4] Ensemble diagnostics with {args.ensemble_members} members ...")
    ens = compute_ensemble_diagnostics(
        pipeline, loader, args.val_batches, device,
        args.max_lon_shift, args.num_inference_steps,
        args.ensemble_members,
    )
    print(f"  Ensemble mean MSE={ens.mean_mse:.6f}  RMSE={ens.mean_rmse:.6f}")
    print(f"  Spread={ens.spread_mean:.6f}  Spread/Skill={ens.spread_skill_ratio:.4f}")
    print(f"  CRPS(approx)={ens.crps_approx:.6f}")
    for k, m in enumerate(ens.member_mse_list):
        print(f"    member {k}: MSE={m:.6f}")

    # --- Build report ---
    report = DetailedEvalReport(
        checkpoint=args.checkpoint,
        model_name=model_name,
        step=step,
        num_val_samples=n_val_samples,
        overall_mse=overall_mse,
        overall_rmse=overall_rmse,
        overall_mae=overall_mae,
        per_channel=ch_metrics,
        spatial_error=spatial,
        inference_ablation=ablation,
        ensemble_diagnostics=ens,
        config={
            "latent_path": args.latent_path,
            "latent_norm_json": args.latent_norm_json,
            "valid_start_time": args.valid_start_time,
            "valid_end_time": args.valid_end_time,
            "input_seq_len": args.input_seq_len,
            "return_seq_len": args.return_seq_len,
            "batch_size": args.batch_size,
            "val_batches": args.val_batches,
            "num_inference_steps": args.num_inference_steps,
            "max_lon_shift": args.max_lon_shift,
            "ensemble_members": args.ensemble_members,
            "seed": args.seed,
        },
    )

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Model: {model_name}  |  Step: {step}  |  Checkpoint: {args.checkpoint}")
    print(f"Overall  MSE={overall_mse:.6f}  RMSE={overall_rmse:.6f}  MAE={overall_mae:.6f}")
    print(f"Ensemble({ens.num_members})  Mean-MSE={ens.mean_mse:.6f}  "
          f"Spread={ens.spread_mean:.6f}  S/S={ens.spread_skill_ratio:.4f}  "
          f"CRPS={ens.crps_approx:.6f}")
    print()
    print("Inference ablation:")
    print(f"  {'mode':<20s} {'MSE':>10s} {'RMSE':>10s} {'MAE':>10s} {'sec/batch':>10s}")
    for r in ablation:
        print(f"  {r.mode:<20s} {r.mse:>10.6f} {r.rmse:>10.6f} {r.mae:>10.6f} "
              f"{r.seconds_per_batch:>10.3f}")
    print("=" * 72)

    # --- Save JSON ---
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
