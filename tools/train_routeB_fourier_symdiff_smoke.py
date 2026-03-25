"""
End-to-end smoke test for Fourier SymDiff training pipeline.

Tests both fourier_haar and fourier_stochastic modes with minimal config.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_tiny_latent_dataset(root: Path) -> tuple[Path, Path]:
    latent_path = root / "latent_smoke.zarr"
    norm_path = root / "latent_norm.json"

    time = np.array(np.arange("2020-01-01T00", "2020-01-01T10", dtype="datetime64[h]"))
    channel = np.arange(4)
    lat = np.arange(8)
    lon = np.arange(16)
    values = np.random.default_rng(0).standard_normal(
        (len(time), len(channel), len(lat), len(lon))
    ).astype(np.float32)

    ds = xr.Dataset(
        data_vars={"latent": (("time", "channel", "lat", "lon"), values)},
        coords={"time": time, "channel": channel, "lat": lat, "lon": lon},
    )
    ds.to_zarr(latent_path, mode="w", consolidated=False)

    mean = values.mean(axis=(0, 2, 3)).astype(np.float32)
    std = values.std(axis=(0, 2, 3)).astype(np.float32)
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": np.clip(std, 1e-6, None).tolist()}, f)

    return latent_path, norm_path


def run_smoke(symmetry_mode: str, run_name: str, latent_path: Path, norm_path: Path, checkpoint_dir: Path) -> None:
    cmd = [
        sys.executable,
        "tools/train_routeB_symdiff.py",
        "--run_name", run_name,
        "--latent_path", str(latent_path),
        "--latent_norm_json", str(norm_path),
        "--train_start_time", "2020-01-01T00",
        "--train_end_time", "2020-01-01T05",
        "--valid_start_time", "2020-01-01T04",
        "--valid_end_time", "2020-01-01T09",
        "--input_seq_len", "1",
        "--return_seq_len", "1",
        "--interval_between_pred", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--max_steps", "2",
        "--hidden_channels", "16",
        "--num_blocks", "2",
        "--time_embed_dim", "32",
        "--num_train_timesteps", "16",
        "--num_inference_steps", "4",
        "--max_lon_shift", "2",
        "--log_every", "1",
        "--val_every", "1",
        "--val_batches", "1",
        "--save_every", "1",
        "--checkpoint_dir", str(checkpoint_dir),
        "--device", "cpu",
        "--symmetry_mode", symmetry_mode,
    ]
    if symmetry_mode == "fourier_stochastic":
        cmd += ["--gamma_warmup_steps", "1", "--gamma_warmup_mode", "haar"]

    print(f"\n--- Running smoke test: {symmetry_mode} ---")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    expected = [
        checkpoint_dir / f"{run_name}_best.pt",
        checkpoint_dir / f"{run_name}_latest.pt",
        checkpoint_dir / f"{run_name}_final.pt",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for {symmetry_mode}: {missing}")
    print(f"[PASS] {symmetry_mode} smoke test passed")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="routeB_fourier_smoke_") as tmpdir:
        tmp_root = Path(tmpdir)
        latent_path, norm_path = build_tiny_latent_dataset(tmp_root)

        # Test fourier_haar
        ckpt_dir_haar = tmp_root / "ckpts_haar"
        run_smoke("fourier_haar", "fourier_haar_smoke", latent_path, norm_path, ckpt_dir_haar)

        # Test fourier_stochastic (with gamma warmup)
        ckpt_dir_stoch = tmp_root / "ckpts_stoch"
        run_smoke("fourier_stochastic", "fourier_stoch_smoke", latent_path, norm_path, ckpt_dir_stoch)

        # Test augmentation
        ckpt_dir_aug = tmp_root / "ckpts_aug"
        run_smoke("augmentation", "aug_smoke", latent_path, norm_path, ckpt_dir_aug)

    print("\n=== All Fourier/augmentation smoke tests passed ===")


if __name__ == "__main__":
    main()
