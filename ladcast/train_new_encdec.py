import argparse
import math
import os
import json
import random
import numpy as np
import torch
import ray
import tempfile  # Added for checkpoint creation
import shutil
import warnings

from diffusers.training_utils import EMAModel
from ray import train
from ray.train import ScalingConfig, Checkpoint, RunConfig, CheckpointConfig # Added Configs
from ray.train.torch import TorchTrainer
from torch.optim import AdamW
from omegaconf import OmegaConf
from huggingface_hub import upload_folder
import wandb

from ladcast.models.DCAE import AutoencoderDC
from ladcast.models.utils import get_scheduler_with_min_lr
from ladcast.dataloader.ray_dataloader import get_zarr_timestamps, ZarrLazyMapper
#from ladcast.dataloader.weather_dataset import weather_dataset_preprocess_batch
from ladcast.dataloader.utils import precompute_mean_std, get_static_conditioning_tensor
from ladcast.metric.utils import process_tensor_for_loss
from ladcast.utils import instantiate_from_config
from ladcast.evaluate.utils import get_normalized_lat_weights_based_on_cos

def train_func(config):
    # Unpack config
    args = config["args"]
    general_config = config["general_config"]
    train_config = config["train_config"]
    data_config = config["data_config"]
    encdec_config = config["encdec_config"]
    optimizer_config = config["optimizer_config"]
    loss_fn_config = config["loss_fn_config"]
    loss_scale_config = config["loss_scale_config"]
    ema_config = config["ema_config"]
    normalization_param_dict = config["normalization_param_dict"]
    lr_scheduler_config = config.get("lr_scheduler_config", None)

    PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    model = AutoencoderDC.from_config(config=encdec_config)
        
    if args["ft_decoder"]:
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(True)
        
    model = ray.train.torch.prepare_model(model)

    if args.get("compile", False):
        print("Compiling AR model with torch.compile()...")
        model = torch.compile(model)
    
    optimizer = AdamW(model.parameters(), **optimizer_config)
    
    loss_fn = instantiate_from_config(loss_fn_config)
    
    device = train.torch.get_device()

    rank = train.get_context().get_world_rank()
    worker_seed = train_config["seed"] + rank
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    mixed_precision = train_config.get("mixed_precision", "no") 
    amp_dtype = None
    
    if mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        raise NotImplementedError("FP16 mixed precision is not supported in this script due to potential instability.")
    else:
        warnings.warn(f"Mixed precision not set or unrecognized ({mixed_precision}), defaulting to fp32.")
        amp_dtype = torch.float32

    print(f"[Rank {rank}] Training with precision: {mixed_precision}")

    if lr_scheduler_config is not None:
        # LR Scheduler
        global_batch_size = train_config["batch_size"] * train_config["num_gpus"]
        
        # math.ceil to account for the final partial batch
        num_update_steps_per_epoch = math.ceil(data_config["num_train_samples"] / global_batch_size)
        max_train_steps = train_config["num_train_epochs"] * num_update_steps_per_epoch

        lr_scheduler = get_scheduler_with_min_lr(
            lr_scheduler_config["name"],
            optimizer=optimizer,
            base_lr=optimizer_config["lr"],
            min_lr=lr_scheduler_config.get("min_lr", 0.0),
            num_warmup_steps=lr_scheduler_config["num_warmup_steps"],
            num_training_steps=max_train_steps,
            num_cycles=lr_scheduler_config.get("num_cycles", 1),
            power=lr_scheduler_config.get("power", 1.0),
        )
    
    ema_model= None
    if ema_config["use_ema"]:
        # We access model.module because 'model' is wrapped in DDP by prepare_model
        ema_model = EMAModel(
            model.parameters(), # DDP parameters
            decay=ema_config["ema_max_decay"],
            use_ema_warmup=True,
            update_after_step=ema_config["ema_update_after_step"],
            inv_gamma=ema_config["ema_inv_gamma"],
            power=ema_config["ema_power"],
            model_cls=AutoencoderDC,
            model_config=model.module.config, 
            foreach=ema_config.get("foreach", False), 
        )
        if ema_config["offload_ema"]:
            ema_model.to("cpu")
        else:
            ema_model.to(device)

    # --- Precompute Normalization Statistics ---
    # Dynamic variables
    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, variable_names=data_config["atmospheric_vars"] + data_config["surface_vars"]
    )
    mean_tensor = mean_tensor[:, None, None].to(device)
    std_tensor = std_tensor[:, None, None].to(device)

    # Static variables (Needed for unnormalization during validation metric calc)
    static_mean_tensor, static_std_tensor = precompute_mean_std(
        normalization_param_dict, variable_names=data_config["static_names"]
    )
    static_mean_tensor = static_mean_tensor[:, None, None].to(device)
    static_std_tensor = static_std_tensor[:, None, None].to(device)
    
    # Combined stats for metric calculation (Dynamic + Static)
    full_mean_tensor = torch.cat((mean_tensor, static_mean_tensor), dim=0)
    full_std_tensor = torch.cat((std_tensor, static_std_tensor), dim=0)
    
    # Latitude weights
    latitude = np.linspace(-88.5, 90, 120) # crop south pole
    lat_weight = get_normalized_lat_weights_based_on_cos(latitude)
    lat_weight = torch.from_numpy(lat_weight).to(device).unsqueeze(1) # (lat, 1)

    static_conditioning_tensor = config["static_conditioning_tensor"].to(device)


    # Get dataset shards
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    wandb_enabled = args["wandb"]
    if wandb_enabled and rank ==0:
            wandb.init(
                project="DCAE_ray",
                config=config,
                group="2019cutoff",
                name=general_config["output_dir"].split("/")[-1],
                reinit=True 
            )
    
    # --- Checkpoint Loading Logic ---
    start_epoch = 0
    global_step = 0
    
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            print(f"[Rank {rank}] Loading checkpoint from directory: {checkpoint_dir}")
            
            model_path = os.path.join(checkpoint_dir, "model.pt")
            if os.path.exists(model_path):
                model_state = torch.load(model_path, map_location=device)
                if train_config["num_gpus"] > 1:
                    model.module.load_state_dict(model_state)
                else:
                    model.load_state_dict(model_state)
                del model_state

            optim_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optim_path):
                optim_state = torch.load(optim_path, map_location=device)
                optimizer.load_state_dict(optim_state)
                del optim_state
            else:
                warnings.warn(f"[Rank {rank}] No optimizer state found in checkpoint.")

            ema_path = os.path.join(checkpoint_dir, "ema.pt")
            if ema_model is not None and os.path.exists(ema_path):
                print(f"[Rank {rank}] Loading EMA state...")
                ema_state = torch.load(ema_path, map_location="cpu") # Load to CPU first to save GPU mem
                ema_model.load_state_dict(ema_state)
                if not ema_config["offload_ema"]:
                    ema_model.to(device)
                del ema_state

            scheduler_path = os.path.join(checkpoint_dir, "lr_scheduler.pt")
            if os.path.exists(scheduler_path) and lr_scheduler_config is not None:
                scheduler_state = torch.load(scheduler_path, map_location=device)
                lr_scheduler.load_state_dict(scheduler_state)
                del scheduler_state

            state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(state_path):
                state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
                
                start_epoch = state_dict["epoch"] + 1
                global_step = state_dict["global_step"]
                
                # Restore Random States
                torch.set_rng_state(state_dict["rng_torch"])
                torch.cuda.set_rng_state_all(state_dict["rng_cuda"])
                np.random.set_state(state_dict["rng_numpy"])
                random.setstate(state_dict["rng_python"])
                
                print(f"[Rank {rank}] Resumed epoch {start_epoch}, RNG states restored.")

    # Training Loop
    for epoch in range(start_epoch, train_config["num_train_epochs"]):
        # --- TRAINING ---
        model.train()
        for batch in train_ds.iter_torch_batches(batch_size=train_config["batch_size"],
                                                 prefetch_batches=train_config.get("prefetch_batches", 1)):
            # batch is a dict: {"data": tensor, "timestamp": tensor, "nan_mask": tensor}
            x = batch["data"].to(device)
            nan_mask = batch["nan_mask"].to(device)
            
            # Preprocess
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and mixed_precision != "no")):
                #x, nan_mask = weather_dataset_preprocess_batch(x, mean_tensor, std_tensor, sst_channel_idx=data_config["sst_channel_idx"])
                static_expanded = static_conditioning_tensor.expand(
                    x.size(0), -1, -1, -1
                ).clone()  # (B, C_static, lat, lon)            

                lat_weight_expanded = lat_weight.expand(
                    x.size(0), model.module.config.out_channels, -1, -1
                ).clone()  # (B, C, lat, 1)

                dec = model(x, return_static=True, static_conditioning_tensor=static_expanded).sample

                # mask values corresponding to nan in SST
                dec, x = process_tensor_for_loss(
                    dec, x, nan_mask, sst_channel_idx=data_config["sst_channel_idx"]
                )
                # static reconstruction loss for return_static=True
                x = torch.cat((x, static_expanded), dim=1)

                # Loss
                if train_config.get("lat_weighted_loss", False):
                    loss_fn_loss = loss_fn(
                        dec, x, weight=lat_weight_expanded
                    )
                else:
                    loss_fn_loss = loss_fn(dec, x)
                    
                loss = loss_scale_config["loss_fn_scale"] * loss_fn_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler_config is not None:
                lr_scheduler.step()
            
            # --- EMA Step ---
            if ema_model is not None:
                if ema_config["offload_ema"]:
                    ema_model.to(device=device, non_blocking=True)
                
                ema_model.step(model.parameters())
                
                if ema_config["offload_ema"]:
                    ema_model.to(device="cpu", non_blocking=True)
                    
            global_step += 1

            if wandb_enabled and rank == 0:
                logs = {
                    "train/step_recon_loss": loss_fn_loss.item(),
                    #"train/global_step": global_step,
                    "train/epoch": epoch
                }
                if ema_model is not None:
                    logs["train/ema_decay"] = ema_model.cur_decay_value
                if lr_scheduler_config is not None:
                    logs["train/lr"] = lr_scheduler.get_last_lr()[0]
                wandb.log(logs, step=global_step)
        
        avg_val_loss = None
        mean_val_rmse = None
        
        if train_config["val_every_epochs"] > 0 and (epoch + 1) % train_config["val_every_epochs"] == 0:
            print(f"Starting validation at epoch {epoch}..\n")
            
            # 1. Store original parameters and load EMA weights
            if ema_model is not None:
                ema_model.store(model.parameters())
                ema_model.copy_to(model.parameters())
            
            model.eval()
            val_loss_sum = 0.0
            val_lw_mse_sum = 0.0
            total_val_samples = 0
            
            with torch.no_grad():
                for batch in val_ds.iter_torch_batches(batch_size=train_config["batch_size"],
                                                       prefetch_batches=train_config.get("prefetch_batches", 1)):
                    x = batch["data"].to(device)
                    nan_mask = batch["nan_mask"].to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and mixed_precision != "no")):
                        #x, nan_mask = weather_dataset_preprocess_batch(x, mean_tensor, std_tensor, sst_channel_idx=data_config["sst_channel_idx"])
                        static_expanded = static_conditioning_tensor.expand(x.size(0), -1, -1, -1).clone()

                        dec = model(x, return_static=True, static_conditioning_tensor=static_expanded).sample
                        dec, x = process_tensor_for_loss(dec, x, nan_mask, sst_channel_idx=data_config["sst_channel_idx"])
                        x = torch.cat((x, static_expanded), dim=1)
                        
                        if train_config.get("lat_weighted_loss", False):
                            lat_weight_expanded_val = lat_weight.expand(x.size(0), model.module.config.out_channels, -1, -1)
                            loss_fn_loss_val = loss_fn(dec, x, weight=lat_weight_expanded_val)
                        else:
                            loss_fn_loss_val = loss_fn(dec, x)

                    val_loss_sum += loss_fn_loss_val.item() * x.size(0)
                    total_val_samples += x.size(0)

                    # Metrics (unnormalized RMSE with latitude weights)
                    pred_un = dec * full_std_tensor + full_mean_tensor
                    targ_un = x * full_std_tensor + full_mean_tensor
                    mse = (pred_un - targ_un) ** 2
                    lw_mse = mse * lat_weight.unsqueeze(0).unsqueeze(0)
                    batch_lw_mse = lw_mse.mean(dim=(2, 3)).sum(dim=0)
                    val_lw_mse_sum += batch_lw_mse

            # 2. Restore original parameters after validation
            if ema_model is not None:
                ema_model.restore(model.parameters())

            # Distributed Aggregation
            scalar_tensor = torch.tensor([val_loss_sum, total_val_samples], device=device)
            torch.distributed.all_reduce(scalar_tensor, op=torch.distributed.ReduceOp.SUM)
            global_val_loss_sum = scalar_tensor[0].item()
            global_total_samples = scalar_tensor[1].item()

            if not isinstance(val_lw_mse_sum, torch.Tensor):
                val_lw_mse_sum = torch.tensor(val_lw_mse_sum, device=device)
            
            torch.distributed.all_reduce(val_lw_mse_sum, op=torch.distributed.ReduceOp.SUM)
            
            avg_val_loss = global_val_loss_sum / global_total_samples
            avg_lw_rmse = torch.sqrt(val_lw_mse_sum / global_total_samples)
            mean_val_rmse = avg_lw_rmse.mean().item()

            if wandb_enabled and rank == 0:
                log_dict = {
                    "val/recon_loss": avg_val_loss,
                    "val/lw_rmse_avg": mean_val_rmse,
                    "epoch": epoch
                }
                
                current_idx = 0
                for var_name in data_config["atmospheric_vars"]:
                    for level in PRESSURE_LEVELS: 
                        log_dict[f"val/lw_rmse_{var_name}_lvl{level}"] = avg_lw_rmse[current_idx].item()
                        current_idx += 1
                for var_name in data_config["surface_vars"]:
                    log_dict[f"val/lw_rmse_{var_name}"] = avg_lw_rmse[current_idx].item()
                    current_idx += 1
                for var_name in data_config["static_names"]:
                    log_dict[f"val/lw_rmse_{var_name}"] = avg_lw_rmse[current_idx].item()
                    current_idx += 1
                
                wandb.log(log_dict)

        save_every_epochs = train_config.get("save_every_epochs", train_config["val_every_epochs"])
        if (save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0) or (epoch == train_config["num_train_epochs"] - 1):
            # --- Checkpointing with EMA ---
            checkpoint = None
            temp_checkpoint_dir = None
            if rank == 0:
                temp_checkpoint_dir = tempfile.mkdtemp()
                
                # Save independent files to prevent OOM on load
                torch.save(model.module.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                
                if ema_model is not None:
                    torch.save(ema_model.state_dict(), os.path.join(temp_checkpoint_dir, "ema.pt"))

                if lr_scheduler_config is not None:
                    torch.save(lr_scheduler.state_dict(), os.path.join(temp_checkpoint_dir, "lr_scheduler.pt"))

                # Save Metadata + RNG States
                training_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(),
                    "rng_numpy": np.random.get_state(),
                    "rng_python": random.getstate(),
                }
                torch.save(training_state, os.path.join(temp_checkpoint_dir, "training_state.pt"))
                
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    
            train.report({
                "epoch": epoch,
                "global_step": global_step,
                "val_recon_loss": avg_val_loss, # TODO: raise err if save_every_epochs != val_every_epochs
                "val_lw_rmse": mean_val_rmse
            }, checkpoint=checkpoint)

            if rank == 0 and temp_checkpoint_dir:
                shutil.rmtree(temp_checkpoint_dir)

    # Save model in diffusers safetensor format
    if rank == 0:
        if ema_model is not None:
            print("Swapping EMA weights into model for upload...")
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        
        final_save_path = os.path.join(general_config.get("output_dir", "./ray_results"), "encdec")
        os.makedirs(final_save_path, exist_ok=True)
        
        print(f"Saving final model to {final_save_path}...")
        if train_config["num_gpus"] > 1:
            model.module.save_pretrained(final_save_path)
        else:
            model.save_pretrained(final_save_path)

        if ema_model is not None:
            ema_model.restore(model.parameters())

    # --- PUSH TO HUB LOGIC (Executed Once at End of Training) ---
    if args["push_to_hub"] and rank == 0:
        print("Training finished. Preparing to push to Hugging Face Hub...")
        repo_name = args["hub_model_id"]
        path_in_repo = general_config["output_dir"].split("/")[-1]
        
        print(f"Uploading to Hub Repository: {repo_name}, Path: {path_in_repo}")
        
        upload_folder(
            repo_id=repo_name,
            folder_path=final_save_path,
            path_in_repo=path_in_repo,
            commit_message="End of training generation (Ray Train)",
            ignore_patterns=["checkpoint_", "*.pt"], # Ignore raw checkpoints if they exist nearby
            token=args["hub_token"]
        )
        print("Upload complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusers model with Ray.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--encdec_cls", type=str, default="dcae", help="The encdec model class to use.")
    parser.add_argument("--ft_decoder", action="store_true", help="Whether to fine-tune the decoder.")
    parser.add_argument("--resume_from_ckpt", action="store_true", help="Resume from the last checkpoint in output_dir.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model with torch.compile().")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load Config
    config = OmegaConf.load(args.config)
    encdec_config = OmegaConf.to_container(config.pop("encdec", OmegaConf.create()), resolve=True)
    optimizer_config = OmegaConf.to_container(config.pop("optimizer", OmegaConf.create()), resolve=True)
    loss_fn_config = OmegaConf.to_container(config.pop("loss_fn", OmegaConf.create()), resolve=True)
    loss_scale_config = OmegaConf.to_container(config.pop("loss_scale", OmegaConf.create()), resolve=True)
    general_config = OmegaConf.to_container(config.pop("general", OmegaConf.create()), resolve=True)
    data_config = OmegaConf.to_container(config.pop("data", OmegaConf.create()), resolve=True)
    train_config = OmegaConf.to_container(config.pop("train", OmegaConf.create()), resolve=True)
    ema_config = OmegaConf.to_container(config.pop("ema", OmegaConf.create()), resolve=True)
    lr_scheduler_config = OmegaConf.to_container(config.pop("lr_scheduler", OmegaConf.create()), resolve=True)

    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"RAY_DEDUP_LOGS": "0"}},
            object_store_memory=general_config.get("ray_object_store_memory", 8) * 1024 * 1024 * 1024 
        )
    
    with open(data_config["norm_json_path"], "r") as f:
        normalization_param_dict = json.load(f)
        
    mean_tensor_cpu, std_tensor_cpu = precompute_mean_std(
        normalization_param_dict, variable_names=data_config["atmospheric_vars"] + data_config["surface_vars"]
    )
    # Reshape for broadcasting (C, 1, 1)
    mean_tensor_cpu = mean_tensor_cpu[:, None, None]
    std_tensor_cpu = std_tensor_cpu[:, None, None]
    
    # Construct static conditioning tensor
    static_conditioning_tensor = get_static_conditioning_tensor(
        data_config["zarr_path"],
        static_names=data_config["static_names"],
        crop_south_pole=True,
        normalize=True,
    )
    print(f"Static conditioning tensor shape: {static_conditioning_tensor.shape}\n")

    # order: atm vars first (each 13 pressure lvls), then surface vars
    sst_channel_idx = len(data_config["atmospheric_vars"]) * 13 + data_config["surface_vars"].index("sea_surface_temperature") 
    print(f"Sea surface temperature channel index: {sst_channel_idx}\n")
    data_config["sst_channel_idx"] = sst_channel_idx # added to data_config
    
    # --- Data Loading ---
    train_timestamps = get_zarr_timestamps(data_config["zarr_path"], start=train_config["train_start"], end=train_config["train_end"])
    print(f"Found {len(train_timestamps)} training samples, starting: {train_timestamps[0]['timestamp']}, ending: {train_timestamps[-1]['timestamp']}\n")
    data_config["num_train_samples"] = len(train_timestamps)
    
    train_ds = ray.data.from_items(train_timestamps)
    train_ds = train_ds.random_shuffle(seed=train_config["seed"])
    train_ds = train_ds.map_batches(
        ZarrLazyMapper,
        fn_constructor_args=(
            data_config["zarr_path"], 
            data_config["surface_vars"], 
            data_config["atmospheric_vars"],
            True,               # preprocess=True
            mean_tensor_cpu,    # mean
            std_tensor_cpu,     # std
            sst_channel_idx,    # sst_idx
            True,               # crop_south_pole
            True                # incl_sur_pressure
        ),
        batch_size=train_config.get("map_batches_batch_size", None),
        compute=ray.data.ActorPoolStrategy(size=train_config.get("actor_pool_size", None)),
        num_cpus=train_config.get("map_batches_num_cpus", 1),
        batch_format="default",
    )
    
    val_timestamps = get_zarr_timestamps(data_config["zarr_path"], start=train_config["val_start"], end=train_config["val_end"])
    print(f"Found {len(val_timestamps)} validation samples, starting: {val_timestamps[0]['timestamp']}, ending: {val_timestamps[-1]['timestamp']}\n")

    val_ds = ray.data.from_items(val_timestamps)
    # map_batches is needed to load
    val_ds = val_ds.map_batches(
        ZarrLazyMapper,
        fn_constructor_args=(
            data_config["zarr_path"], 
            data_config["surface_vars"], 
            data_config["atmospheric_vars"],
            True,               # preprocess=True
            mean_tensor_cpu,    # mean
            std_tensor_cpu,     # std
            sst_channel_idx,    # sst_idx
            True,               # crop_south_pole
            True                # incl_sur_pressure
        ),
        batch_size=train_config.get("map_batches_batch_size", None),
        compute=ray.data.ActorPoolStrategy(size=train_config.get("actor_pool_size", None)),
        num_cpus=train_config.get("map_batches_num_cpus", 1),
        batch_format="default",
    )
    
    # --- Configure Output and Checkpointing ---
    output_dir = os.path.abspath(general_config.get("output_dir", "./ray_results"))
    experiment_name = general_config.get("experiment_name", "dcae_train_run")
    
    run_config = RunConfig(
        storage_path=output_dir,
        name=experiment_name,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
        )
    )

    trainer_args = {
        "train_loop_per_worker": train_func,
        "train_loop_config": {
            "args": vars(args),
            "general_config": general_config,
            "train_config": train_config,
            "data_config": data_config,
            "encdec_config": encdec_config,
            "optimizer_config": optimizer_config,
            "loss_fn_config": loss_fn_config,
            "loss_scale_config": loss_scale_config,
            "ema_config": ema_config,
            "lr_scheduler_config": lr_scheduler_config if lr_scheduler_config else None,
            "normalization_param_dict": normalization_param_dict,
            "static_conditioning_tensor": static_conditioning_tensor,
        },
        "scaling_config": ScalingConfig(num_workers=train_config["num_gpus"], use_gpu=True),
        "run_config": run_config
    }
    
    # --- Logic to Resume or Start Fresh ---
    if not args.resume_from_ckpt:
        # If directory exists but we asked for fresh start, consider appending time or checking
        experiment_path = os.path.join(output_dir, experiment_name)
        if os.path.exists(experiment_path):
            warnings.warn(f"Warning: experiment_name '{experiment_name}' exists but resume_from_ckpt is False. Ray might implicitly load the checkpoint if we don't change the name.")

    trainer = TorchTrainer(datasets={"train": train_ds, "val": val_ds}, **trainer_args)
    
    print("Starting training...")
    result = trainer.fit()
    print(f"Training result: {result}")

if __name__ == "__main__":
    main()