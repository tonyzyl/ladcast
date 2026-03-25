import argparse
import json
import math
import os
import random
import sys
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
from ladcast.models.routeB_fourier_shift import FourierLongitudeShift, LearnablePhaseGamma
from ladcast.models.routeB_symdiff_denoiser import RouteBSymDiffDenoiser
from ladcast.models.symmetry import sample_lon_roll_shifts
from ladcast.pipelines.pipeline_routeB_symdiff import RouteBSymDiffPipeline

# Symmetry modes and their model names
_MODE_TO_MODEL_NAME = {
    "identity": "routeB_diffusion_nonsymm",
    "stochastic": "routeB_symdiff",
    "augmentation": "routeB_diffusion_aug",
    "fourier_haar": "routeB_symdiff_fourier_haar",
    "fourier_stochastic": "routeB_symdiff_fourier",
}

# Which symmetry_type the denoiser should use
_MODE_TO_SYMMETRY_TYPE = {
    "identity": "roll",
    "stochastic": "roll",
    "augmentation": "roll",
    "fourier_haar": "fourier",
    "fourier_stochastic": "fourier",
}

# Default valid inference mode per symmetry_mode
_MODE_TO_DEFAULT_VALID_INFERENCE = {
    "identity": "identity",
    "stochastic": "random_single",
    "augmentation": "identity",
    "fourier_haar": "fourier_random_single",
    "fourier_stochastic": "fourier_random_single",
}

ALL_SYMMETRY_MODES = list(_MODE_TO_MODEL_NAME.keys())
ALL_VALID_INFERENCE_MODES = [
    "identity", "random_single", "fixed_group", "group_mean",
    "fourier_identity", "fourier_random_single", "fourier_mean", "fourier_grid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train minimal RouteB conditional latent diffusion with optional stochastic symmetrisation."
    )
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
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--time_embed_dim", type=int, default=256)
    parser.add_argument("--max_lon_shift", type=int, default=16)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--symmetry_mode", type=str, choices=ALL_SYMMETRY_MODES, default="stochastic")
    parser.add_argument("--valid_inference_mode", type=str, choices=ALL_VALID_INFERENCE_MODES, default=None,
                        help="Inference mode for validation. Auto-selected based on symmetry_mode if not set.")
    parser.add_argument("--valid_fixed_shift", type=int, default=0)
    parser.add_argument("--valid_num_symmetry_samples", type=int, default=1)
    # Gamma warmup (Task 4)
    parser.add_argument("--gamma_warmup_steps", type=int, default=0,
                        help="Freeze gamma net for this many steps (fourier_stochastic only)")
    parser.add_argument("--gamma_warmup_mode", type=str, default="haar", choices=["identity", "haar"],
                        help="Phase sampling during warmup: identity=no shift, haar=uniform random")
    parser.add_argument("--gamma_hidden_dim", type=int, default=256)
    parser.add_argument("--gamma_noise_dim", type=int, default=16)
    # General
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="~/ladcast/checkpoints/routeB_diffusion")
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


def emit_log(**payload) -> None:
    print(json.dumps(payload, sort_keys=True))


def default_run_name(args: argparse.Namespace) -> str:
    model_name = _MODE_TO_MODEL_NAME.get(args.symmetry_mode, "routeB_diffusion")
    return f"{model_name}_in{args.input_seq_len}_out{args.return_seq_len}_bs{args.batch_size}_steps{args.max_steps}_seed{args.seed}"


def checkpoint_path(checkpoint_dir: str, run_name: str, suffix: str) -> str:
    return os.path.join(checkpoint_dir, f"{run_name}_{suffix}.pt")


def check_batch_shapes(batch: dict, input_seq_len: int, return_seq_len: int) -> None:
    x_in = batch["x_in"]
    x_out = batch["x_out"]
    if x_in.ndim != 5 or x_out.ndim != 5:
        raise RuntimeError("Expected x_in/x_out to be 5D tensors")
    if x_in.shape[1] != input_seq_len or x_out.shape[1] != return_seq_len:
        raise RuntimeError(f"Sequence mismatch: x_in={tuple(x_in.shape)} x_out={tuple(x_out.shape)}")


def create_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------------------------
# Training group state samplers
# ---------------------------------------------------------------------------

def sample_training_group_state(batch_size: int, device: torch.device, max_lon_shift: int, symmetry_mode: str) -> torch.Tensor | None:
    """Sample group state for roll-based modes (identity / stochastic). Used by compare script too."""
    if symmetry_mode == "identity" or max_lon_shift <= 0:
        return None
    if symmetry_mode == "stochastic":
        return sample_lon_roll_shifts(batch_size, max_lon_shift, device)
    raise ValueError(f"Unknown symmetry_mode for roll group state: {symmetry_mode}")


def _sample_fourier_phase(batch_size: int, device: torch.device, mode: str) -> torch.Tensor | None:
    """Sample phase angles for fourier modes."""
    if mode == "identity":
        return None
    if mode == "haar":
        return torch.rand(batch_size, device=device) * 2 * math.pi
    raise ValueError(f"Unknown fourier phase mode: {mode}")


# ---------------------------------------------------------------------------
# Diffusion train step (unified for all modes)
# ---------------------------------------------------------------------------

def diffusion_train_step(
        model: RouteBSymDiffDenoiser,
        scheduler: DDPMScheduler,
        cond: torch.Tensor,
        x0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_lon_shift: int,
        symmetry_mode: str,
        # Fourier-specific
        gamma_net: LearnablePhaseGamma | None = None,
        fourier_shifter: FourierLongitudeShift | None = None,
        step: int = 0,
        gamma_warmup_steps: int = 0,
        gamma_warmup_mode: str = "haar",
) -> float:
    B = cond.shape[0]
    device = cond.device

    # --- Augmentation mode: roll data, then train without group_state ---
    if symmetry_mode == "augmentation":
        W = cond.shape[-1]
        k = torch.randint(0, W, (1,)).item()
        cond = torch.roll(cond, shifts=k, dims=-1)
        x0 = torch.roll(x0, shifts=k, dims=-1)
        group_state = None
    # --- Roll-based modes ---
    elif symmetry_mode in ("identity", "stochastic"):
        group_state = sample_training_group_state(B, device, max_lon_shift, symmetry_mode)
    # --- Fourier Haar (no learnable gamma) ---
    elif symmetry_mode == "fourier_haar":
        group_state = torch.rand(B, device=device) * 2 * math.pi
    # --- Fourier stochastic (learnable gamma with warmup) ---
    elif symmetry_mode == "fourier_stochastic":
        if step < gamma_warmup_steps:
            # Warmup: freeze gamma, use simple phase
            if gamma_net is not None:
                for p in gamma_net.parameters():
                    p.requires_grad_(False)
            group_state = _sample_fourier_phase(B, device, gamma_warmup_mode)
        else:
            # Full recursive symmetrisation
            if gamma_net is not None:
                for p in gamma_net.parameters():
                    p.requires_grad_(True)
                phi_0 = torch.rand(B, device=device) * 2 * math.pi
                # Inverse-shift input by phi_0 for gamma prediction
                cond_4d = cond.reshape(B, -1, cond.shape[-2], cond.shape[-1])
                cond_pre = fourier_shifter.inverse_shift(cond_4d, phi_0)
                delta = gamma_net(cond_pre)
                group_state = phi_0 + delta
            else:
                group_state = torch.rand(B, device=device) * 2 * math.pi
    else:
        raise ValueError(f"Unknown symmetry_mode: {symmetry_mode}")

    eps = torch.randn_like(x0)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
    noisy = scheduler.add_noise(x0, eps, timesteps)

    optimizer.zero_grad(set_to_none=True)
    pred_eps = model(cond, noisy, timesteps, group_state=group_state)
    loss = F.mse_loss(pred_eps.float(), eps.float())
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite train loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def validate_by_sampling(
        model: RouteBSymDiffDenoiser,
        scheduler: DDPMScheduler,
        val_loader: DataLoader,
        val_batches: int,
        device: torch.device,
        max_lon_shift: int,
        num_inference_steps: int,
        inference_mode: str,
        fixed_shift: int,
        num_symmetry_samples: int,
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
        x0 = batch["x_out"].to(device)
        x0_hat = pipeline(
            cond,
            num_inference_steps=num_inference_steps,
            max_lon_shift=max_lon_shift,
            inference_mode=inference_mode,
            fixed_shift=fixed_shift,
            num_symmetry_samples=num_symmetry_samples,
        )
        latent_mse = F.mse_loss(x0_hat.float(), x0.float())
        if not torch.isfinite(latent_mse):
            raise RuntimeError(f"Non-finite validation latent MSE: {latent_mse.item()}")
        losses.append(float(latent_mse.item()))

    model.train()
    return float(np.mean(losses))


def save_checkpoint(
        ckpt_path: str,
        model,
        optimizer,
        step: int,
        train_loss: float,
        valid_latent_mse: float,
        args: argparse.Namespace,
        channels: int,
        spatial: tuple[int, int],
        run_name: str,
        model_name: str,
        gamma_net=None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "train_eps_mse": train_loss,
        "valid_latent_mse": valid_latent_mse,
        "run_name": run_name,
        "model_name": model_name,
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
            "hidden_channels": args.hidden_channels,
            "num_blocks": args.num_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "time_embed_dim": args.time_embed_dim,
            "max_lon_shift": args.max_lon_shift,
            "num_train_timesteps": args.num_train_timesteps,
            "num_inference_steps": args.num_inference_steps,
            "beta_schedule": args.beta_schedule,
            "symmetry_mode": args.symmetry_mode,
            "symmetry_type": _MODE_TO_SYMMETRY_TYPE.get(args.symmetry_mode, "roll"),
            "valid_inference_mode": args.valid_inference_mode,
            "valid_fixed_shift": args.valid_fixed_shift,
            "valid_num_symmetry_samples": args.valid_num_symmetry_samples,
            "gamma_warmup_steps": args.gamma_warmup_steps,
            "gamma_warmup_mode": args.gamma_warmup_mode,
        },
        "channels": channels,
        "spatial": spatial,
    }
    if gamma_net is not None:
        payload["gamma_net_state_dict"] = gamma_net.state_dict()
    torch.save(payload, ckpt_path)


def build_dataset(args: argparse.Namespace, split: str) -> RouteBLatentDataset:
    if split == "train":
        start_time = args.train_start_time
        end_time = args.train_end_time
    elif split == "valid":
        start_time = args.valid_start_time
        end_time = args.valid_end_time
    else:
        raise ValueError(f"Unknown split: {split}")

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.input_seq_len != 1 or args.return_seq_len != 1:
        raise ValueError("This minimal strict diffusion implementation only supports input_seq_len=1 and return_seq_len=1.")
    if args.interval_between_pred != 1:
        raise ValueError("This minimal strict diffusion implementation only supports single-step future latent prediction.")
    if args.log_every <= 0 or args.val_every <= 0 or args.save_every <= 0 or args.val_batches <= 0:
        raise ValueError("log/val/save intervals and val_batches must be > 0")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")

    # Auto-select valid inference mode if not set
    if args.valid_inference_mode is None:
        args.valid_inference_mode = _MODE_TO_DEFAULT_VALID_INFERENCE[args.symmetry_mode]

    if args.valid_inference_mode == "group_mean" and args.valid_num_symmetry_samples <= 0:
        raise ValueError("valid_num_symmetry_samples must be > 0 for group_mean")

    run_name = args.run_name or default_run_name(args)
    model_name = _MODE_TO_MODEL_NAME.get(args.symmetry_mode, "routeB_diffusion")
    symmetry_type = _MODE_TO_SYMMETRY_TYPE.get(args.symmetry_mode, "roll")
    device = torch.device(args.device)
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = build_dataset(args, split="train")
    valid_dataset = build_dataset(args, split="valid")
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_loader = create_dataloader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    sample0 = train_dataset[0]
    check_batch_shapes({"x_in": sample0["x_in"].unsqueeze(0), "x_out": sample0["x_out"].unsqueeze(0)}, args.input_seq_len, args.return_seq_len)
    channels = int(sample0["x_in"].shape[1])
    spatial = (int(sample0["x_in"].shape[2]), int(sample0["x_in"].shape[3]))

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon",
        clip_sample=False,
    )
    model = RouteBSymDiffDenoiser(
        channels=channels,
        cond_seq_len=args.input_seq_len,
        target_seq_len=args.return_seq_len,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        time_embed_dim=args.time_embed_dim,
        symmetry_type=symmetry_type,
    ).to(device)

    # Gamma net for fourier_stochastic mode
    gamma_net = None
    fourier_shifter = None
    all_params = list(model.parameters())
    if args.symmetry_mode == "fourier_stochastic":
        gamma_net = LearnablePhaseGamma(
            input_channels=args.input_seq_len * channels,
            hidden_dim=args.gamma_hidden_dim,
            noise_dim=args.gamma_noise_dim,
        ).to(device)
        fourier_shifter = FourierLongitudeShift()
        all_params += list(gamma_net.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    start_step = 1
    best_valid_latent_mse = float("inf")
    latest_valid_latent_mse = float("nan")
    if args.resume_from is not None:
        ckpt = torch.load(os.path.expanduser(args.resume_from), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if gamma_net is not None and "gamma_net_state_dict" in ckpt:
            gamma_net.load_state_dict(ckpt["gamma_net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0)) + 1
        best_valid_latent_mse = float(ckpt.get("valid_latent_mse", float("inf")))
        latest_valid_latent_mse = float(ckpt.get("valid_latent_mse", float("nan")))

    emit_log(
        event="init",
        model_name=model_name,
        run_name=run_name,
        train_len=len(train_dataset),
        valid_len=len(valid_dataset),
        channels=channels,
        spatial=spatial,
        symmetry_mode=args.symmetry_mode,
        symmetry_type=symmetry_type,
        valid_inference_mode=args.valid_inference_mode,
        gamma_warmup_steps=args.gamma_warmup_steps,
    )
    model.train()
    if gamma_net is not None:
        gamma_net.train()
    train_iter = infinite_loader(train_loader)
    train_loss = float("nan")

    for step in range(start_step, args.max_steps + 1):
        batch = next(train_iter)
        check_batch_shapes(batch, args.input_seq_len, args.return_seq_len)
        cond = batch["x_in"].to(device)
        x0 = batch["x_out"].to(device)
        train_loss = diffusion_train_step(
            model, scheduler, cond, x0, optimizer,
            args.max_lon_shift, args.symmetry_mode,
            gamma_net=gamma_net,
            fourier_shifter=fourier_shifter,
            step=step,
            gamma_warmup_steps=args.gamma_warmup_steps,
            gamma_warmup_mode=args.gamma_warmup_mode,
        )

        if step % args.log_every == 0 or step == 1:
            emit_log(event="train", model_name=model_name, run_name=run_name, step=step, train_eps_mse=train_loss)

        if step % args.val_every == 0:
            latest_valid_latent_mse = validate_by_sampling(
                model,
                scheduler,
                valid_loader,
                args.val_batches,
                device,
                args.max_lon_shift,
                args.num_inference_steps,
                args.valid_inference_mode,
                args.valid_fixed_shift,
                args.valid_num_symmetry_samples,
            )
            emit_log(
                event="valid",
                model_name=model_name,
                run_name=run_name,
                step=step,
                valid_latent_mse=latest_valid_latent_mse,
                valid_inference_mode=args.valid_inference_mode,
                valid_num_symmetry_samples=args.valid_num_symmetry_samples,
            )
            if latest_valid_latent_mse < best_valid_latent_mse:
                best_valid_latent_mse = latest_valid_latent_mse
                best_path = checkpoint_path(checkpoint_dir, run_name, "best")
                save_checkpoint(best_path, model, optimizer, step, train_loss, latest_valid_latent_mse, args, channels, spatial, run_name, model_name, gamma_net)
                emit_log(event="checkpoint", model_name=model_name, run_name=run_name, step=step, checkpoint_type="best", path=best_path)

        if step % args.save_every == 0:
            step_path = checkpoint_path(checkpoint_dir, run_name, f"step_{step:06d}")
            latest_path = checkpoint_path(checkpoint_dir, run_name, "latest")
            save_checkpoint(step_path, model, optimizer, step, train_loss, latest_valid_latent_mse, args, channels, spatial, run_name, model_name, gamma_net)
            save_checkpoint(latest_path, model, optimizer, step, train_loss, latest_valid_latent_mse, args, channels, spatial, run_name, model_name, gamma_net)
            emit_log(event="checkpoint", model_name=model_name, run_name=run_name, step=step, checkpoint_type="step", path=step_path)
            emit_log(event="checkpoint", model_name=model_name, run_name=run_name, step=step, checkpoint_type="latest", path=latest_path)

    final_path = checkpoint_path(checkpoint_dir, run_name, "final")
    save_checkpoint(final_path, model, optimizer, args.max_steps, train_loss, latest_valid_latent_mse, args, channels, spatial, run_name, model_name, gamma_net)
    emit_log(event="checkpoint", model_name=model_name, run_name=run_name, step=args.max_steps, checkpoint_type="final", path=final_path)


if __name__ == "__main__":
    main()
