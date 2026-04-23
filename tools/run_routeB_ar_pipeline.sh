@@ -31,31 +31,32 @@ echo "=============================="

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
  --model_type symdiff_resnet \
  --checkpoint_path ~/ladcast/checkpoints/routeB_ar_smoke.pt


echo "=============================="
echo "Pipeline finished"
echo "=============================="