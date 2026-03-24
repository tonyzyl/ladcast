import argparse
import sys
from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.models.routeB_symdiff import RouteBSymDiffDenoiser
from ladcast.models.symmetry import sample_lon_roll_shifts
from ladcast.pipelines.pipeline_routeB_symdiff import RouteBSymDiffPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent smoke test for the strict RouteB SymDiff path.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_seq_len", type=int, default=1)
    parser.add_argument("--return_seq_len", type=int, default=1)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--num_train_timesteps", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--max_lon_shift", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_seq_len != 1 or args.return_seq_len != 1:
        raise ValueError("The strict RouteB SymDiff smoke test only supports seq len 1->1.")

    device = torch.device(args.device)
    cond = torch.randn(
        args.batch_size,
        args.input_seq_len,
        args.channels,
        args.height,
        args.width,
        device=device,
    )
    target = torch.randn(
        args.batch_size,
        args.return_seq_len,
        args.channels,
        args.height,
        args.width,
        device=device,
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    denoiser = RouteBSymDiffDenoiser(
        channels=args.channels,
        cond_seq_len=args.input_seq_len,
        target_seq_len=args.return_seq_len,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        time_embed_dim=args.time_embed_dim,
    ).to(device)

    denoiser.train()
    eps = torch.randn_like(target)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
    noisy = scheduler.add_noise(target, eps, timesteps)
    shifts = sample_lon_roll_shifts(args.batch_size, args.max_lon_shift, device) if args.max_lon_shift > 0 else None
    pred_eps = denoiser(cond, noisy, timesteps, group_state=shifts)
    assert pred_eps.shape == target.shape, pred_eps.shape
    loss = (pred_eps - eps).square().mean()
    assert torch.isfinite(loss), loss
    loss.backward()

    denoiser.eval()
    pipeline = RouteBSymDiffPipeline(denoiser=denoiser, scheduler=scheduler)
    sample = pipeline(
        cond,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(0),
        max_lon_shift=args.max_lon_shift,
    )
    assert sample.shape == target.shape, sample.shape
    assert torch.isfinite(sample).all(), "Generated sample contains non-finite values"

    print("routeB symdiff smoke passed")


if __name__ == "__main__":
    main()
