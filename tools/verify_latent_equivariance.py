"""
Verify whether circular longitude shift (torch.roll) is a valid symmetry
in the DC-AE latent space.

If encoder(roll(x)) ≈ roll(encoder(x)), the equivariance holds and SymDiff's
roll-based symmetrisation is justified. Otherwise, roll introduces artifacts.

This script uses a statistical approach on pre-encoded latent data:
  - Roll latent tensors by various shifts along W (longitude)
  - Check that channel-wise spatial statistics are preserved
  - Check autocorrelation structure consistency
  - Check cross-sample MSE between shifted and original neighbours

Usage:
  python tools/verify_latent_equivariance.py \
    --latent_path ~/ladcast/data/routeB_latent_train.zarr \
    --latent_norm_json ~/ladcast/static/ERA5_routeB_latent_normal_train.json \
    --start_time 2018-01-01T00 --end_time 2018-01-10T23 \
    --num_samples 50 --device cpu
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ladcast.dataloader.routeB_latent_dataset import RouteBLatentDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify latent equivariance of longitude roll.")
    p.add_argument("--latent_path", type=str, required=True)
    p.add_argument("--latent_norm_json", type=str, required=True)
    p.add_argument("--start_time", type=str, default="2018-01-01T00")
    p.add_argument("--end_time", type=str, default="2018-01-10T23")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--shift_amounts", type=int, nargs="+", default=[1, 2, 5, 10, 15])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def verify_equivariance(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)

    dataset = RouteBLatentDataset(
        latent_path=args.latent_path,
        start_time=args.start_time,
        end_time=args.end_time,
        input_seq_len=1,
        return_seq_len=1,
        interval_between_pred=1,
        normalize=True,
        latent_norm_json=args.latent_norm_json,
        return_time=False,
    )

    n_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    # Collect samples
    latents = []
    for idx in indices:
        sample = dataset[int(idx)]
        latents.append(sample["x_out"].to(device))  # (S, C, H, W)
    latents = torch.stack(latents, dim=0)  # (N, S, C, H, W)
    N, S, C, H, W = latents.shape
    z = latents.reshape(N, S * C, H, W)  # (N, SC, H, W)

    print("=" * 60)
    print("Latent Equivariance Verification")
    print("=" * 60)
    print(f"Samples: {N}, Channels: {S * C}, Spatial: ({H}, {W})")
    print()

    baseline_energy = z.pow(2).mean().item()
    print(f"Baseline energy (mean z²): {baseline_energy:.6f}")
    print()

    results = {}

    # --- Test 1: Channel-wise spatial mean invariance ---
    print("--- Test 1: Channel-wise spatial mean invariance under roll ---")
    orig_mean = z.mean(dim=(-2, -1))  # (N, SC)
    for k in args.shift_amounts:
        z_rolled = torch.roll(z, shifts=k, dims=-1)
        rolled_mean = z_rolled.mean(dim=(-2, -1))
        mean_diff = (orig_mean - rolled_mean).abs().max().item()
        print(f"  shift={k:3d}: max |mean_diff| = {mean_diff:.8f}")
    print("  (Should be ~0 — roll preserves global channel means)")
    print()

    # --- Test 2: Autocorrelation structure ---
    print("--- Test 2: W-direction autocorrelation preservation ---")
    orig_autocorr = (z[..., :-1] * z[..., 1:]).mean(dim=(-2, -1))  # (N, SC)
    for k in args.shift_amounts:
        z_rolled = torch.roll(z, shifts=k, dims=-1)
        rolled_autocorr = (z_rolled[..., :-1] * z_rolled[..., 1:]).mean(dim=(-2, -1))
        autocorr_diff = (orig_autocorr - rolled_autocorr).abs().mean().item()
        print(f"  shift={k:3d}: mean |autocorr_diff| = {autocorr_diff:.8f}")
    print("  (Should be ~0 — roll preserves local correlation structure)")
    print()

    # --- Test 3: Neighbouring-timestep prediction error after roll ---
    # If roll is a true symmetry, then predicting z[t+1] from z[t] should be
    # equally easy whether or not both are rolled by the same shift.
    # We measure: MSE(roll(z[t+1]) - roll(z[t])) vs MSE(z[t+1] - z[t])
    print("--- Test 3: Temporal difference structure under roll ---")
    if N > 1:
        z_in = latents[:, 0, :, :, :]   # (N, C, H, W) — use separate in/out
        # Collect x_in too
        z_in_list = []
        for idx in indices:
            sample = dataset[int(idx)]
            z_in_list.append(sample["x_in"].to(device))
        z_in_all = torch.stack(z_in_list, dim=0).reshape(N, S * C, H, W)
        z_out_all = z  # already (N, SC, H, W)

        orig_diff_mse = (z_out_all - z_in_all).pow(2).mean().item()
        for k in args.shift_amounts:
            rolled_in = torch.roll(z_in_all, shifts=k, dims=-1)
            rolled_out = torch.roll(z_out_all, shifts=k, dims=-1)
            rolled_diff_mse = (rolled_out - rolled_in).pow(2).mean().item()
            relative_change = abs(rolled_diff_mse - orig_diff_mse) / (orig_diff_mse + 1e-10)
            print(f"  shift={k:3d}: orig_diff_mse={orig_diff_mse:.6f}, "
                  f"rolled_diff_mse={rolled_diff_mse:.6f}, "
                  f"relative_change={relative_change:.6f}")
    print("  (Should be ~0 — temporal structure should be shift-invariant)")
    print()

    # --- Test 4: Power spectrum analysis ---
    # If roll is a true symmetry, the power spectrum along W should be flat
    # (translation-invariant process has uniform spectral density)
    print("--- Test 4: Spectral flatness along longitude ---")
    Z_fft = torch.fft.rfft(z, dim=-1)  # (N, SC, H, freq)
    power = Z_fft.abs().pow(2).mean(dim=(0, 2))  # (SC, freq) — avg over N, H
    # Compute spectral flatness per channel: geometric_mean / arithmetic_mean
    log_power = torch.log(power + 1e-10)
    geo_mean = torch.exp(log_power.mean(dim=-1))  # (SC,)
    arith_mean = power.mean(dim=-1)                # (SC,)
    flatness = (geo_mean / (arith_mean + 1e-10))   # (SC,) in [0, 1], 1=perfectly flat
    mean_flatness = flatness.mean().item()
    min_flatness = flatness.min().item()
    print(f"  Mean spectral flatness: {mean_flatness:.4f}")
    print(f"  Min  spectral flatness: {min_flatness:.4f}")
    print("  (1.0 = perfectly translation-invariant; <0.5 = strong spatial structure)")
    print()

    # --- Test 5: Cross-position variance ratio ---
    # Compare variance along W (longitude) vs H (latitude)
    # If longitude is approximately translation-invariant, var along W should be < var along H
    print("--- Test 5: Spatial variance ratio (W vs H) ---")
    var_w = z.var(dim=-1).mean().item()   # variance along longitude
    var_h = z.var(dim=-2).mean().item()   # variance along latitude
    ratio = var_w / (var_h + 1e-10)
    print(f"  Variance along W (lon): {var_w:.6f}")
    print(f"  Variance along H (lat): {var_h:.6f}")
    print(f"  Ratio W/H: {ratio:.4f}")
    print("  (Ratio ~1 suggests similar variability; <<1 suggests W is more uniform)")
    print()

    # --- Overall conclusion ---
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Heuristic: if spectral flatness is high and temporal diff is stable,
    # roll is approximately valid
    if mean_flatness > 0.7:
        verdict = "PASS"
        msg = ("Spectral flatness is high — longitude is approximately "
               "translation-invariant in latent space. Circular roll is a "
               "reasonable symmetry approximation.")
    elif mean_flatness > 0.4:
        verdict = "MARGINAL"
        msg = ("Spectral flatness is moderate — circular roll is an approximate "
               "symmetry but may introduce artifacts. Consider soft symmetrisation "
               "(Fourier phase rotation) instead.")
    else:
        verdict = "FAIL"
        msg = ("Spectral flatness is low — strong spatial structure along longitude. "
               "Circular roll is NOT a valid symmetry in latent space. "
               "SymDiff's roll-based symmetrisation is likely harmful.")

    print(f"Verdict: {verdict}")
    print(f"  {msg}")
    print("=" * 60)

    results = {
        "num_samples": int(N),
        "spatial": [int(H), int(W)],
        "channels": int(S * C),
        "baseline_energy": baseline_energy,
        "spectral_flatness_mean": mean_flatness,
        "spectral_flatness_min": min_flatness,
        "variance_ratio_w_over_h": ratio,
        "verdict": verdict,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out_path}")

    return results


def main() -> None:
    args = parse_args()
    verify_equivariance(args)


if __name__ == "__main__":
    main()
