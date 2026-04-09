# QM9

To run SymDiff

```
python main_qm9.py --exp_name symdiff --model dit_dit_gaussian_dynamics --dataset qm9 --datadir YOUR_DATADIR \
                   --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --diffusion_noise_schedule polynomial_2 \
                   --n_epochs 4350 --batch_size 256 --lr 2e-4 --clipping_type norm --max_grad_norm 2.0 --ema_decay 0.9999 \
                   --weight_decay 1e-12 --use_amsgrad --normalize_factors [1,4,10] \
                   --n_stability_samples 500 --test_epochs 20 --wandb_usr YOUR_WANDB_USERNAME --save_model True \
                   --xh_hidden_size 184 --K 184 \
                   --enc_hidden_size 128 --enc_depth 8 --enc_num_heads 4 --enc_mlp_ratio 4.0 --dec_hidden_features 64 \
                   --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4.0 --mlp_dropout 0.0 \
                   --noise_dims 3 --noise_std 1.0 \
                   --mlp_type swiglu
```

To run SymDiff*

```
python main_qm9.py --exp_name symdiff_star --model dit_dit_gaussian_dynamics --dataset qm9 --datadir YOUR_DATADIR \
                   --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --diffusion_noise_schedule polynomial_2 \
                   --n_epochs 4350 --batch_size 256 --lr 1e-4 --clipping_type norm --max_grad_norm 2.0 --ema_decay 0.9999 \
                   --weight_decay 1e-12 --use_amsgrad --normalize_factors [1,4,10] \
                   --n_stability_samples 500 --test_epochs 30 --wandb_usr YOUR_WANDB_USERNAME --save_model True \
                   --xh_hidden_size 382 --K 382 \
                   --enc_hidden_size 216 --enc_depth 10 --enc_num_heads 8 --enc_mlp_ratio 4.0 --dec_hidden_features 80 \
                   --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4.0 --mlp_dropout 0.0 \
                   --noise_dims 3 --noise_std 1.0 \
                   --mlp_type swiglu
```                  


To run DiT

```
python main_qm9.py --exp_name dit --model dit_gaussian_dynamics --dataset qm9 --datadir YOUR_DATADIR \
                   --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --diffusion_noise_schedule polynomial_2 \
                   --n_epochs 4350 --batch_size 256 --lr 2e-4 --clipping_type norm --max_grad_norm 2.0 --ema_decay 0.9999 \
                   --weight_decay 1e-12 --use_amsgrad --normalize_factors [1,4,10] \
                   --n_stability_samples 500 --test_epochs 20 --wandb_usr YOUR_WANDB_USERNAME --save_model True \
                   --xh_hidden_size 184 --K 184 \
                   --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4.0 --mlp_dropout 0.0 \
                   --mlp_type swiglu
```

To run DiT-Aug, just add the option `--data_augmentation True` to the above command.

