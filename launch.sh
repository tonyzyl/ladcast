#!/usr/bin/env bash
set -euo pipefail

# Strict read-only source-data launch template for LaDCast.
# Goal: NEVER write into source dataset paths; only write to OUTPUT_ROOT.
#
# Usage:
#   bash scripts/readonly_source_data_launch_template.sh
#
# Before running:
# 1) Ensure SOURCE_ERA5_ZARR and LATENT_ZARR are mounted/readable.
# 2) Prefer RO mount for source paths at OS/container level.
# 3) Set ACCELERATE_CONFIG to your actual accelerate config.

############################
# User configurable section
############################
SOURCE_ERA5_ZARR="/data_large/zarr_datasets/ERA5_1_5_1h_zarr_conservative_1979-2024"
LATENT_ZARR="~/ladcast/data"  # precomputed latents
OUTPUT_ROOT="~/ladcast/data/readonly_run_$(date +%Y%m%d_%H%M%S)"
ACCELERATE_CONFIG="~/ladcast/data/default.yaml"

# Optional evaluation inputs
CLIMATOLOGY_ZARR="/data_large/zarr_dataset/climatology.zarr"
ENCDEC_MODEL_NAME="V0.1.X/DCAE"

############################
# Safety guards
############################
require_exists() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] Missing required path: $p" >&2
    exit 1
  fi
}

require_not_writable() {
  local p="$1"
  if [[ -w "$p" ]]; then
    echo "[FATAL] Path is writable but must be read-only for this template: $p" >&2
    echo "        Please mount/chmod source path as read-only before running." >&2
    exit 1
  fi
}

require_exists "$SOURCE_ERA5_ZARR"
require_exists "$LATENT_ZARR"
require_exists "$ACCELERATE_CONFIG"

# Strict-mode policy: fail if source paths are writable.
require_not_writable "$SOURCE_ERA5_ZARR"
require_not_writable "$LATENT_ZARR"

mkdir -p "$OUTPUT_ROOT"/{configs,logs,checkpoints,predictions,metrics}

############################
# Build runtime config copy
############################
RUNTIME_CONFIG="$OUTPUT_ROOT/configs/ladcast_375M.readonly.yaml"
python - "$RUNTIME_CONFIG" "$LATENT_ZARR" "$OUTPUT_ROOT/checkpoints" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

out_cfg = Path(sys.argv[1])
latent_zarr = sys.argv[2]
out_dir = sys.argv[3]

cfg = OmegaConf.load("ladcast/configs/ladcast_375M.yaml")
cfg.train_dataloader.ds_path = latent_zarr
cfg.train_dataloader.load_in_memory = False
cfg.general.output_dir = out_dir
cfg.general.tracker_project_name = "ladcast_readonly"
cfg.general.logging_dir = "logs"
# keep symmetry default disabled unless user explicitly edits this runtime config.

out_cfg.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, out_cfg)
print(f"Wrote runtime config: {out_cfg}")
PY

############################
# 1) AR training (reads latent zarr, writes checkpoints only)
############################
accelerate launch --config_file "$ACCELERATE_CONFIG" \
  ladcast/train_AR.py \
  --config "$RUNTIME_CONFIG" \
  --ar_cls transformer \
  --encdec_cls dcae \
  --lat_weighted_loss \
  --checkpoints_total_limit 5 \
  2>&1 | tee "$OUTPUT_ROOT/logs/train_ar.log"

# Pick latest checkpoint folder automatically.
LATEST_CKPT_DIR=$(find "$OUTPUT_ROOT/checkpoints" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [[ -z "${LATEST_CKPT_DIR:-}" ]]; then
  echo "[FATAL] No checkpoint-* found in $OUTPUT_ROOT/checkpoints" >&2
  exit 1
fi

############################
# 2) Rollout prediction (reads source zarr, writes predictions)
############################
accelerate launch --config_file "$ACCELERATE_CONFIG" \
  ladcast/evaluate/pred_rollout.py \
  --data_path "$SOURCE_ERA5_ZARR" \
  --encdec_model_name "$ENCDEC_MODEL_NAME" \
  --ar_model_path "$LATEST_CKPT_DIR/ar_model" \
  --start_date 2018-01-01T00:00:00 \
  --end_date 2018-12-31T12:00:00 \
  --ensemble_size 20 \
  --num_inference_steps 20 \
  --return_seq_len 4 \
  --total_lead_time_hour 240 \
  --step_size_hour 6 \
  --input_seq_len 1 \
  --latent_normal_json ladcast/static/ERA5_latent_normal_1979_2017_lat84.json \
  --normalization_json ladcast/static/ERA5_normal_1979_2017.json \
  --lsm_path ladcast/static/240x121_land_sea_mask.pt \
  --orography_path ladcast/static/240x121_orography.pt \
  --save_as_latent \
  --sampler_type edm \
  --batch_size 8 \
  --output "$OUTPUT_ROOT/predictions" \
  2>&1 | tee "$OUTPUT_ROOT/logs/pred_rollout.log"

############################
# 3) Ensemble evaluation (reads source+predictions, writes metrics)
############################
if [[ -e "$CLIMATOLOGY_ZARR" ]]; then
  accelerate launch --config_file "$ACCELERATE_CONFIG" \
    ladcast/evaluate/evaluate_ens_gpu.py \
    --data_path "$SOURCE_ERA5_ZARR" \
    --climatology_path "$CLIMATOLOGY_ZARR" \
    --result_path "$OUTPUT_ROOT/predictions" \
    --encdec_model "$ENCDEC_MODEL_NAME" \
    --start_date 2018-01-01T00:00:00 \
    --end_date 2019-01-16T00:00:00 \
    --total_lead_time_hour 240 \
    --step_size_hour 6 \
    --normalization_json ladcast/static/ERA5_normal_1979_2017.json \
    --crop_init \
    --output "$OUTPUT_ROOT/metrics" \
    2>&1 | tee "$OUTPUT_ROOT/logs/evaluate_ens.log"
else
  echo "[WARN] CLIMATOLOGY_ZARR not found; skip evaluate_ens_gpu stage." | tee -a "$OUTPUT_ROOT/logs/evaluate_ens.log"
fi

echo "[DONE] Read-only source-data pipeline finished."
echo "       Outputs are in: $OUTPUT_ROOT"
