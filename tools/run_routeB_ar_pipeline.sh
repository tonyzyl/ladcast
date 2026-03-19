#!/bin/bash
set -e

LATENT_PATH=~/ladcast/data/routeB_latent_eval.zarr
LATENT_NORM_JSON=~/ladcast/static/ERA5_routeB_latent_normal_1979_2017.json
CHECKPOINT_PATH=~/ladcast/checkpoints/routeB_ar_smoke.pt

echo "=============================="
echo "1) Checking latent dataset"
echo "=============================="

python tools/check_routeB_latent.py \
  --latent_path $LATENT_PATH \
  --max_samples 5000


echo "=============================="
echo "2) Computing latent normalization"
echo "=============================="

python tools/compute_routeB_latent_norm.py \
  --latent_path $LATENT_PATH \
  --start_time 1979-01-01 \
  --end_time 2017-12-31 \
  --output_json $LATENT_NORM_JSON


echo "=============================="
echo "3) Testing latent dataset"
echo "=============================="

python tools/test_routeB_latent_dataset.py \
  --latent_path ~/ladcast/data/routeB_latent_eval.zarr \
  --latent_norm_json ~/ladcast/static/ERA5_routeB_latent_normal_1979_2017.json \
  --start_time 1979-02-01T06:00:00 \
  --end_time 1979-02-03T05:00:00 \
  --input_seq_len 1 \
  --return_seq_len 1 \
  --interval_between_pred 1


echo "=============================="
echo "4) AR smoke test"
echo "=============================="

python tools/train_routeB_ar_smoke.py \
  --latent_path ~/ladcast/data/routeB_latent_eval.zarr \
  --latent_norm_json ~/ladcast/static/ERA5_routeB_latent_normal_1979_2017.json \
  --start_time 1979-02-01T06:00:00 \
  --end_time 1979-02-04T05:00:00 \
  --input_seq_len 1 \
  --return_seq_len 1 \
  --interval_between_pred 1 \
  --batch_size 2 \
  --max_steps 20 \
  --checkpoint_path ~/ladcast/checkpoints/routeB_ar_smoke.pt


echo "=============================="
echo "Pipeline finished"
echo "=============================="