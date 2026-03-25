#!/usr/bin/env bash
# =============================================================================
# Step 3: Detailed evaluation of trained SymDiff checkpoint
# =============================================================================
# Usage:
#   bash scripts/run_routeB_step3_detailed_eval.sh [GPU_ID] [CHECKPOINT]
#
# If CHECKPOINT is not provided, uses the best checkpoint from Step 1 default run.
# =============================================================================
set -euo pipefail

GPU_ID="${1:-2}"
# Default checkpoint: best from default run name
DEFAULT_CKPT="$HOME/ladcast/checkpoints/routeB_diffusion/routeB_symdiff_in1_out1_bs8_steps20000_seed42_best.pt"
CKPT="${2:-$DEFAULT_CKPT}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================"
echo " Step 3: Detailed SymDiff Evaluation"
echo " GPU: $GPU_ID"
echo " Checkpoint: $CKPT"
echo " Start: $(date)"
echo "========================================"

python tools/eval_routeB_symdiff_detailed.py \
  --checkpoint "$CKPT" \
  --latent_path ~/ladcast/data/routeB_latent_train.zarr \
  --latent_norm_json ~/ladcast/static/ERA5_routeB_latent_normal_train.json \
  --valid_start_time 2018-02-01T00 \
  --valid_end_time 2018-02-07T23 \
  --input_seq_len 1 \
  --return_seq_len 1 \
  --interval_between_pred 1 \
  --batch_size 8 \
  --num_workers 4 \
  --val_batches 20 \
  --num_inference_steps 50 \
  --max_lon_shift 16 \
  --ensemble_members 8 \
  --output_json tmp/routeB_symdiff_detailed_eval.json \
  --device cuda

echo ""
echo "========================================"
echo " Step 3 DONE: $(date)"
echo " Results: tmp/routeB_symdiff_detailed_eval.json"
echo "========================================"
