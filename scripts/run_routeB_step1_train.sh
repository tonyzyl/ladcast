#!/usr/bin/env bash
# =============================================================================
# Step 1: Train RouteB SymDiff model
# =============================================================================
# Usage:
#   bash scripts/run_routeB_step1_train.sh [GPU_ID]
#
# Default GPU_ID=2. Override by passing first argument:
#   bash scripts/run_routeB_step1_train.sh 0
# =============================================================================
set -euo pipefail

GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================"
echo " Step 1: Training RouteB SymDiff"
echo " GPU: $GPU_ID"
echo " Start: $(date)"
echo "========================================"

python tools/train_routeB_symdiff.py \
  --symmetry_mode stochastic \
  --valid_inference_mode random_single \
  --latent_path ~/ladcast/data/routeB_latent_train.zarr \
  --latent_norm_json ~/ladcast/static/ERA5_routeB_latent_normal_train.json \
  --train_start_time 2018-01-01T00 \
  --train_end_time 2018-01-31T23 \
  --valid_start_time 2018-02-01T00 \
  --valid_end_time 2018-02-07T23 \
  --input_seq_len 1 \
  --return_seq_len 1 \
  --interval_between_pred 1 \
  --batch_size 8 \
  --num_workers 4 \
  --max_steps 20000 \
  --hidden_channels 128 \
  --num_blocks 6 \
  --kernel_size 3 \
  --time_embed_dim 256 \
  --num_train_timesteps 1000 \
  --num_inference_steps 50 \
  --max_lon_shift 16 \
  --val_every 200 \
  --val_batches 20 \
  --save_every 500 \
  --checkpoint_dir ~/ladcast/checkpoints/routeB_diffusion \
  --device cuda

echo ""
echo "========================================"
echo " Step 1 DONE: $(date)"
echo "========================================"
