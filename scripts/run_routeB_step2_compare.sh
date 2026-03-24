#!/usr/bin/env bash
# =============================================================================
# Step 2: Compare minimal baselines (tiny_ar, non_symm_resnet, diffusion, symdiff)
# =============================================================================
# Usage:
#   bash scripts/run_routeB_step2_compare.sh [GPU_ID]
# =============================================================================
set -euo pipefail

GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================"
echo " Step 2: Comparing RouteB Baselines"
echo " GPU: $GPU_ID"
echo " Start: $(date)"
echo "========================================"

python tools/compare_routeB_minimal_baselines.py \
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
  --max_steps 2000 \
  --val_batches 20 \
  --num_inference_steps 50 \
  --output_json tmp/routeB_minimal_compare.json

echo ""
echo "========================================"
echo " Step 2 DONE: $(date)"
echo " Results: tmp/routeB_minimal_compare.json"
echo "========================================"
