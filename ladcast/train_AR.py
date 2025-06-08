import argparse
import json
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from huggingface_hub import upload_folder
from omegaconf import OmegaConf
from packaging import version
from torch.optim import AdamW
from tqdm.auto import tqdm

from ladcast.dataloader.ar_dataloder import (
    convert_datetime_to_int,
    prepare_ar_dataloader,
)
from ladcast.dataloader.utils import (
    filter_time_range,
    inverse_normalize_transform_3D,
    precompute_mean_std,
)
from ladcast.evaluate.utils import get_crps, get_normalized_lat_weights_based_on_cos
from ladcast.models.DCAE import AutoencoderDC
from ladcast.models.embeddings import convert_int_to_datetime
from ladcast.models.LaDCast_3D_model import LaDCastTransformer3DModel
from ladcast.models.utils import Karras_sigmas_lognormal
from ladcast.pipelines.pipeline_AR import AutoRegressive2DPipeline
from ladcast.pipelines.utils import ensemble_AR_sampler, get_sigmas
from ladcast.utils import flatten_and_filter_config, instantiate_from_config

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(
    phase_name: str,
    xr_dataset: xr.Dataset,
    config,
    ar_model,
    full_field_mean_tensor: torch.Tensor,
    full_field_std_tensor: torch.Tensor,
    input_seq_len: int,
    return_seq_len: int,
    encdec_model,
    latent_transform_func: callable,
    latent_inv_transform_func: callable,
    noise_scheduler_config,
    accelerator: accelerate.Accelerator,
    timestamp_list: list,
    step_size_hour: int = 6,
    total_lead_time_hour: int = 240,
    ensemble_size: int = 10,
    num_inference_steps: int = 20,
    eval_ms: bool = True,
    eval_crps: bool = True,
    return_df: bool = False,
):
    """
    timestamp_list: list of timestamps @ t=0,
    the current detault orientation of compressed latent is lat: from south(-ve) to north(+ve), lon: [0, 360)
    """

    def create_wandb_table(tensor, step_hour_list, col_names) -> wandb.Table:
        """The tensor shall has shape (timestamp, channel)"""
        tensor_data = tensor.tolist() if hasattr(tensor, "tolist") else tensor
        table_data = []

        for i, row in enumerate(tensor_data):
            table_data.append([step_hour_list[i]] + row)

        table = wandb.Table(data=table_data, columns=col_names)

        return table

    def create_pd_dataframe(tensor, step_hour_list, col_names) -> pd.DataFrame:
        """
        The tensor shall have shape (timestamp, channel)
        Returns a pandas DataFrame with the same data structure as the wandb table
        """
        tensor_data = tensor.tolist() if hasattr(tensor, "tolist") else tensor
        data_dict = {"lead time": step_hour_list}

        for col_idx, col_name in enumerate(col_names):
            data_dict[col_name] = [row[col_idx] for row in tensor_data]

        df = pd.DataFrame(data_dict)

        return df

    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    num_atm_vars = 6

    atm_var_names = [
        f"{atm_var}_level{level}"
        for atm_var in config.channel_names[:num_atm_vars]
        for level in levels
    ]
    sur_var_names = config.channel_names[num_atm_vars:]
    col_names = atm_var_names + sur_var_names

    if total_lead_time_hour % step_size_hour != 0:
        raise ValueError("total_lead_time_hour must be divisible by step_size_hour.")
    total_num_steps = int(total_lead_time_hour / step_size_hour)
    num_repetitions = math.ceil(total_num_steps / return_seq_len)

    step_hour_list = [i * step_size_hour for i in range(1, total_num_steps + 1)]

    noise_scheduler = instantiate_from_config(noise_scheduler_config)
    pipeline = AutoRegressive2DPipeline(ar_model, scheduler=noise_scheduler)

    latitude = np.linspace(-88.5, 90, 120)  # crop south pole
    lat_weight = get_normalized_lat_weights_based_on_cos(latitude)  # (lat,)
    lat_weight = torch.from_numpy(lat_weight).to(accelerator.device)  # (lat,)

    # for the metric, we will stroe the single and the ensemble mean
    # TODO, dynamically set the out channels
    edm_single_mse = torch.full(
        (len(timestamp_list), 84, total_num_steps), float("nan"), device="cpu"
    )
    edm_ens_mse = torch.full(
        (len(timestamp_list), 84, total_num_steps), float("nan"), device="cpu"
    )
    if eval_ms:
        ms_single_mse = torch.full(
            (len(timestamp_list), 84, total_num_steps), float("nan"), device="cpu"
        )
        ms_ens_mse = torch.full(
            (len(timestamp_list), 84, total_num_steps), float("nan"), device="cpu"
        )
    if eval_crps:
        edm_crps = torch.full(
            (len(timestamp_list), 84, total_num_steps), float("nan"), device="cpu"
        )
    full_field_mean_tensor = full_field_mean_tensor.to(accelerator.device)
    full_field_std_tensor = full_field_std_tensor.to(accelerator.device)

    for timestamp_idx, init_timestamp in enumerate(timestamp_list):
        # datetime with chronic order as input to the model
        input_datetime_list = [
            init_timestamp - pd.Timedelta(hours=step_size_hour * i)
            for i in range(input_seq_len - 1, -1, -1)
        ]
        known_latents = (
            xr_dataset["latents"].sel(time=input_datetime_list).values
        )  # (T, C, H, W)
        known_latents = torch.from_numpy(known_latents).to(accelerator.device)
        known_latents = known_latents.permute(1, 0, 2, 3)  # -> (C, T, H, W)
        known_latents = latent_transform_func(known_latents)
        known_latents = rearrange(known_latents, "C T H W -> 1 C T H W")

        edm_known_latents = known_latents.clone()
        # print(f"Processing time {init_timestamp} on process {accelerator.process_index}")

        finish_timestamp = init_timestamp + pd.Timedelta(hours=total_lead_time_hour)
        ref_tensor = (
            xr_dataset["latents"]
            .sel(
                time=slice(
                    init_timestamp + pd.Timedelta(hours=step_size_hour),
                    finish_timestamp,
                )
            )
            .values
        )  # (T, C, H, W)
        # print('process index', accelerator.process_index, slice(init_timestamp+pd.Timedelta(hours=step_size_hour), finish_timestamp))
        ref_tensor = torch.from_numpy(ref_tensor)
        # print('process index', accelerator.process_index, ref_tensor.shape)
        ref_tensor = (
            encdec_model.decode(ref_tensor.to(accelerator.device))
            .sample.permute(1, 0, 2, 3)
            .to(accelerator.device)
        )  # (C, T, H, W)
        ref_tensor = inverse_normalize_transform_3D(
            ref_tensor, full_field_mean_tensor, full_field_std_tensor
        )
        # print('process index', accelerator.process_index, ref_tensor.shape, 'start time:', init_timestamp+pd.Timedelta(hours=step_size_hour), 'end time:', finish_timestamp)

        # TODO, dynamically set the out channels
        edm_decoded_tensor = torch.full(
            (ensemble_size, 84, total_num_steps, 120, 240),
            float("nan"),
            device=accelerator.device,
        )
        if eval_ms:
            ms_known_latents = known_latents.clone()
            ms_decoded_tensor = torch.full(
                (ensemble_size, 84, total_num_steps, 120, 240),
                float("nan"),
                device=accelerator.device,
            )
        # rolling out the prediction
        for step in range(num_repetitions):
            # Calculate the lead time for the current prediction step
            # return (ensemble_size, C, T, H, W)
            # print('process', accelerator.process_index, 'step', step, 'of', num_repetitions)
            timestamps = init_timestamp + pd.Timedelta(hours=step * step_size_hour)
            timestamps = convert_datetime_to_int(timestamps)  # -> int format YYYYMMDDHH
            timestamps = torch.tensor([timestamps], device=accelerator.device)  # (1,)
            edm_sample_latents = ensemble_AR_sampler(
                pipeline,
                sample_size=ensemble_size,
                return_seq_len=return_seq_len,
                num_inference_steps=num_inference_steps,
                known_latents=edm_known_latents,
                timestamps=timestamps,
                sampler_type="edm",
                device=pipeline._execution_device,
            )
            # Update known_latents for the next prediction step
            # print('process', accelerator.process_index, 'finish edm_sample_latents')
            edm_known_latents = edm_sample_latents[:, :, -input_seq_len:].clone()

            if eval_ms:
                ms_sample_latents = ensemble_AR_sampler(
                    pipeline,
                    sample_size=ensemble_size,
                    return_seq_len=return_seq_len,
                    num_inference_steps=num_inference_steps,
                    known_latents=ms_known_latents,
                    timestamps=timestamps,
                    sampler_type="pipeline",
                    device=pipeline._execution_device,
                )
                ms_known_latents = ms_sample_latents[:, :, -input_seq_len:].clone()

            for ensemble_idx in range(ensemble_size):
                edm_sample_latents[ensemble_idx] = latent_inv_transform_func(
                    edm_sample_latents[ensemble_idx]
                )  # Rescale VAE latent
                # (T, C, H, W), treat T as batch dim then permute back to (C, T, H, W)
                cur_step = min((step + 1) * return_seq_len, total_num_steps)
                # (ensemble_size, C, T, H, W)
                edm_decoded_tensor[
                    ensemble_idx, :, step * return_seq_len : cur_step
                ] = inverse_normalize_transform_3D(
                    encdec_model.decode(
                        edm_sample_latents[ensemble_idx].permute(1, 0, 2, 3)
                    )
                    .sample.permute(1, 0, 2, 3)
                    .to(accelerator.device),
                    full_field_mean_tensor,
                    full_field_std_tensor,
                )

                if eval_ms:
                    ms_sample_latents[ensemble_idx] = latent_inv_transform_func(
                        ms_sample_latents[ensemble_idx]
                    )
                    ms_decoded_tensor[
                        ensemble_idx, :, step * return_seq_len : cur_step
                    ] = inverse_normalize_transform_3D(
                        encdec_model.decode(
                            ms_sample_latents[ensemble_idx].permute(1, 0, 2, 3)
                        )
                        .sample.permute(1, 0, 2, 3)
                        .to(accelerator.device),
                        full_field_mean_tensor,
                        full_field_std_tensor,
                    )

        edm_single_squared_error = (
            (edm_decoded_tensor - ref_tensor.unsqueeze(0)) ** 2
        ) * lat_weight.view(1, 1, 1, -1, 1)  # (ensemble_size, C, T, H, W)
        edm_ens_squared_error = (
            (edm_decoded_tensor.mean(dim=0) - ref_tensor) ** 2
        ) * lat_weight.view(1, 1, -1, 1)  # (C, T, H, W)
        edm_single_mse[timestamp_idx] = edm_single_squared_error.mean(dim=(0, 3, 4)).to(
            "cpu"
        )  # (C, T)
        edm_ens_mse[timestamp_idx] = edm_ens_squared_error.mean(dim=(2, 3)).to(
            "cpu"
        )  # (C, T)

        if eval_ms:
            ms_single_squared_error = (
                (ms_decoded_tensor - ref_tensor.unsqueeze(0)) ** 2
            ) * lat_weight.view(1, 1, 1, -1, 1)
            ms_ens_squared_error = (
                (ms_decoded_tensor.mean(dim=0) - ref_tensor) ** 2
            ) * lat_weight.view(1, 1, -1, 1)
            ms_single_mse[timestamp_idx] = ms_single_squared_error.mean(
                dim=(0, 3, 4)
            ).to("cpu")
            ms_ens_mse[timestamp_idx] = ms_ens_squared_error.mean(dim=(2, 3)).to("cpu")

        if eval_crps:
            tmp_edm_crps = get_crps(
                edm_decoded_tensor, ref_tensor, ensemble_dim=0
            )  # (C, T, H, W)
            edm_crps[timestamp_idx] = (
                (tmp_edm_crps * lat_weight.view(1, 1, -1, 1)).mean(dim=(2, 3)).to("cpu")
            )  # (C, T)
            crps_col_names = [
                "lead time",
                *[f"CRPS_{col_name}" for col_name in col_names],
            ]

    # test gather within function:
    edm_ens_mse = edm_ens_mse.to(accelerator.device)
    edm_single_mse = edm_single_mse.to(accelerator.device)
    edm_crps = edm_crps.to(accelerator.device)
    # print('process', accelerator.process_index, 'before gather', edm_single_mse.shape, edm_ens_mse.shape)
    edm_ens_mse = accelerator.gather(edm_ens_mse)
    edm_single_mse = accelerator.gather(edm_single_mse)
    edm_crps = accelerator.gather(edm_crps)
    edm_ens_mse = edm_ens_mse.mean(dim=0).to("cpu")
    edm_single_mse = edm_single_mse.mean(dim=0).to("cpu")
    edm_crps = edm_crps.mean(dim=0).to("cpu")

    if eval_ms:
        ms_ens_mse = ms_ens_mse.to(accelerator.device)
        ms_single_mse = ms_single_mse.to(accelerator.device)
        ms_ens_mse = accelerator.gather(ms_ens_mse)
        ms_single_mse = accelerator.gather(ms_single_mse)
        ms_ens_mse = ms_ens_mse.mean(dim=0).to("cpu")
        ms_single_mse = ms_single_mse.mean(dim=0).to("cpu")

    merged_col_names = [
        "lead time",
        *[f"EDM_ens_{col_name}" for col_name in col_names],
        *[f"EDM_single_{col_name}" for col_name in col_names],
    ]
    if eval_ms:
        merged_col_names.extend([f"MS_ens_{col_name}" for col_name in col_names])
        merged_col_names.extend([f"MS_single_{col_name}" for col_name in col_names])
        merged_rmse_tensor = torch.cat(
            [
                torch.sqrt(edm_ens_mse).T,
                torch.sqrt(edm_single_mse).T,
                torch.sqrt(ms_ens_mse).T,
                torch.sqrt(ms_single_mse).T,
            ],
            dim=1,
        )
    else:
        merged_rmse_tensor = torch.cat(
            [torch.sqrt(edm_ens_mse).T, torch.sqrt(edm_single_mse).T], dim=1
        )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            merged_table = create_wandb_table(
                merged_rmse_tensor, step_hour_list, merged_col_names
            )
            if eval_crps:
                crps_table = create_wandb_table(
                    edm_crps.T, step_hour_list, crps_col_names
                )
                tracker.log({"merged_RMSE": merged_table, "CRPS": crps_table})
            else:
                tracker.log({"merged_RMSE": merged_table})

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_df:
        rmse_pd_df = create_pd_dataframe(
            merged_rmse_tensor, step_hour_list, merged_col_names
        )
        if eval_crps:
            crps_pd_df = create_pd_dataframe(edm_crps.T, step_hour_list, crps_col_names)
            return (rmse_pd_df, crps_pd_df)
        else:
            return rmse_pd_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusers model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        default=False,
        help=("If True, load weights only from the checkpoint (not the EMA)."),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store (max number-1)."),
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="tonyzyl/ladcast",
        help="The name of the HF repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--encdec_model",
        type=str,
        default="V0.1.X/DCAE",
        help="The subfolder of the encoder-decoder model in the huggingface Hub.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--ar_cls", type=str, default="unet", help="The Autoregressive model backbone."
    )  # unet, transformer
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--norm_json_path",
        type=str,
        default="static/ERA5_normal_1979_2017.json",
        help=("Path to the JSON file containing the normalization statistics."),
    )
    parser.add_argument(
        "--latent_norm_json_path",
        type=str,
        default="static/ERA5_latent_normal_1979_2017_lat84.json",
        help="Path to the JSON file containing the normalization parameters for the latent space.",
    )
    parser.add_argument(
        "--num_push_forward_steps",
        type=int,
        default=1,
        help="The number of steps to push forward for the model.",
    )
    parser.add_argument(
        "--lat_weighted_loss",
        action="store_true",
        help="Whether to use latitudinally weighted loss.",
    )
    parser.add_argument(
        "--encdec_cls",
        type=str,
        default="dcae",
        help="The Encoder-Decoder model backbone.",
    )  # dcae, vae

    return parser.parse_args()


def main(args):
    # workflow based on: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

    config = OmegaConf.load(args.config)
    tracker_config = flatten_and_filter_config(
        OmegaConf.to_container(config, resolve=True)
    )

    ar_model_config = OmegaConf.to_container(
        config.pop("ar_model", OmegaConf.create()), resolve=True
    )
    noise_scheduler_config = config.pop("noise_scheduler", OmegaConf.create())
    noise_sampler_config = config.pop("noise_sampler", OmegaConf.create())
    accelerator_config = config.pop("accelerator", OmegaConf.create())
    optimizer_config = config.pop("optimizer", OmegaConf.create())
    lr_scheduler_config = config.pop("lr_scheduler", OmegaConf.create())
    train_dataloader_config = config.pop("train_dataloader", OmegaConf.create())
    ema_config = config.pop("ema", OmegaConf.create())
    general_config = config.pop("general", OmegaConf.create())

    with open(args.latent_norm_json_path, "r") as f:
        train_dataloader_config["transform_args"] = json.load(f)
        train_dataloader_config["transform_args"]["target_std"] = 0.5

    if args.ar_cls == "transformer":
        ar_model_cls = LaDCastTransformer3DModel
    else:
        raise NotImplementedError(f"Unknown autoregressive model class: {args.ar_cls}")
    ar_model = ar_model_cls.from_config(config=ar_model_config)
    ar_model.requires_grad_(True)

    repo_name = args.hub_model_id
    if args.encdec_cls == "dcae":
        encdec_cls = AutoencoderDC
    else:
        raise NotImplementedError(
            f"Unknown encoder-decoder model class: {args.encdec_cls}"
        )
    encdec_model = encdec_cls.from_pretrained(
        repo_name, subfolder=args.encdec_model
    )  # set to eval by default
    encdec_model.requires_grad_(False)

    logging_dir = Path(general_config.output_dir, general_config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=general_config.output_dir, logging_dir=logging_dir
    )
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator = Accelerator(
        project_config=accelerator_project_config,
        **OmegaConf.to_container(accelerator_config, resolve=True),
        kwargs_handlers=[kwargs],
    )

    set_seed(general_config.seed)

    with open(args.norm_json_path, "r") as f:
        normalization_param_dict = json.load(f)
        full_field_mean_tensor, full_field_std_tensor = precompute_mean_std(
            normalization_param_dict, general_config.channel_names
        )
        full_field_mean_tensor = full_field_mean_tensor.to("cpu")
        full_field_std_tensor = full_field_std_tensor.to("cpu")

    # Create EMA for the model. Some examples:
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
    if ema_config.use_ema:
        ema_model = EMAModel(
            ar_model.parameters(),
            decay=ema_config.ema_max_decay,
            use_ema_warmup=True,
            update_after_step=ema_config.ema_update_after_step,
            inv_gamma=ema_config.ema_inv_gamma,
            power=ema_config.ema_power,
            model_cls=ar_model_cls,
            model_config=ar_model.config,
            foreach=ema_config.foreach,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if ema_config.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "ar_model_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "ar_model"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def EMA_from_pretrained(path, model_cls) -> "EMAModel":
            # _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
            model_config = model_cls.load_config(path)  # contains ema_kwargs
            model = model_cls.from_pretrained(path)

            ema_model = EMAModel(
                model.parameters(), model_cls=model_cls, model_config=model.config
            )

            ema_model.load_state_dict(model_config)
            return ema_model

        def load_model_hook(models, input_dir):
            if ema_config.use_ema:
                # TODO: follow up on loading checkpoint with EMA, ema_kwargs not properly loaded
                # https://github.com/huggingface/diffusers/discussions/8802
                load_model = EMA_from_pretrained(
                    os.path.join(input_dir, "ar_model_ema"),
                    ar_model_cls,  # , foreach=ema_config.foreach #v0.29.2 does not have foreach
                )
                ema_model.load_state_dict(load_model.state_dict())
                if ema_config.offload_ema:
                    ema_model.pin_memory()
                else:
                    ema_model.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ar_model_cls.from_pretrained(
                    input_dir, subfolder="ar_model"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    noise_scheduler = instantiate_from_config(noise_scheduler_config)
    # inv_noise_scheduler_class = get_inv_noise_scheduler(noise_scheduler_config["scheduler_name"])
    # noise_scheduler = inv_noise_scheduler_class(**noise_scheduler_config["scheduler_params"])

    noise_sampler = Karras_sigmas_lognormal(
        noise_scheduler.sigmas,
        P_mean_start=noise_sampler_config.P_mean_start,
        P_std_start=noise_sampler_config.P_std_start,
        P_mean_end=noise_sampler_config.P_mean_end,
        P_std_end=noise_sampler_config.P_std_end,
    )
    # enhance sampler diversity
    noise_sampler_gen = torch.Generator(device="cpu").manual_seed(
        general_config.seed + accelerator.process_index
    )

    with accelerator.main_process_first():
        # https://github.com/huggingface/accelerate/issues/503
        # https://discuss.huggingface.co/t/shared-memory-in-accelerate/28619
        train_dataloader = prepare_ar_dataloader(**train_dataloader_config)

    assert train_dataloader_config.return_seq_len % args.num_push_forward_steps == 0, (
        f"num_push_forward_steps {args.num_push_forward_steps} must be a divisor of return_seq_len {train_dataloader_config.return_seq_len}"
    )
    num_slice_per_push_forward = int(
        train_dataloader_config.return_seq_len / args.num_push_forward_steps
    )

    val_dataset = xr.open_dataset(
        train_dataloader_config.ds_path, engine="zarr", chunks="auto"
    )
    val_timerange = pd.date_range(start="2017-12-31", end="2019-01-10", freq="6h")
    # val_timerange = pd.date_range(start='2018-12-31', end='2019-03-31', freq='6h') # allowing a 10-day window for covering the pred window
    val_dataset = val_dataset.sel(time=val_timerange)
    val_timerange4pred = list(
        filter_time_range(val_timerange, num_samples_per_month=2, enforce_year="2018")
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if general_config.scale_lr:
        optimizer_config.lr = (
            optimizer_config.lr
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
            * train_dataloader_config.batch_size
        )

    optimizer = AdamW(
        ar_model.parameters(), **OmegaConf.to_container(optimizer_config, resolve=True)
    )

    if "subbatch_steps" in general_config:
        num_subbatch_steps = int(
            general_config.subbatch_steps
        )  # num_steps to augment the batch
    else:
        num_subbatch_steps = 1
    num_warmup_steps_for_scheduler = (
        lr_scheduler_config.num_warmup_steps
        * accelerator.num_processes
        * num_subbatch_steps
    )

    if (
        "num_training_steps" not in general_config
        or general_config.num_training_steps is None
    ):
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        # main update steps per epoch
        num_update_steps_per_epoch = (
            math.ceil(
                len_train_dataloader_after_sharding
                / accelerator.gradient_accumulation_steps
            )
            * num_subbatch_steps
        )
        num_training_steps_for_scheduler = (
            general_config.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = (
            general_config.num_training_steps
            * accelerator.num_processes
            * num_subbatch_steps
        )

    lr_scheduler = get_scheduler(
        lr_scheduler_config.name,
        optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=lr_scheduler_config.num_cycles,
        power=lr_scheduler_config.power,
    )

    if args.gradient_checkpointing:
        ar_model.enable_gradient_checkpointing()

    # TODO: maybe only load vae when logging image
    encdec_model = encdec_model.to(accelerator.device)

    ar_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        ar_model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = (
        math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
        * num_subbatch_steps
    )
    if general_config.num_training_steps is None:
        assert general_config.num_train_epochs is not None, (
            "You need to set either 'num_train_steps' or 'num_train_epochs'"
        )
        general_config.num_training_steps = (
            general_config.num_train_epochs * num_update_steps_per_epoch
        )
        if (
            num_training_steps_for_scheduler
            != general_config.num_training_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    general_config.num_train_epochs = math.ceil(
        general_config.num_training_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        print(tracker_config)
        accelerator.init_trackers(
            general_config.tracker_project_name, config=tracker_config
        )

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        # https://github.com/huggingface/diffusers/issues/6503
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    total_batch_size = (
        train_dataloader_config.batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {general_config.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {train_dataloader_config.batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {general_config.num_training_steps}")
    logger.info(f"  Total training epochs = {general_config.num_train_epochs}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(general_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # if accelerator.is_main_process: # temp fix for only having one random state
            if args.load_weights_only:
                ar_model = ar_model_cls.from_pretrained(
                    os.path.join(general_config.output_dir, path, "ar_model")
                )
                ema_model = EMAModel(
                    ar_model.parameters(),
                    decay=ema_config.ema_max_decay,
                    use_ema_warmup=True,
                    update_after_step=ema_config.ema_update_after_step,
                    inv_gamma=ema_config.ema_inv_gamma,
                    power=ema_config.ema_power,
                    model_cls=ar_model_cls,
                    model_config=ar_model.config,
                    foreach=ema_config.foreach,
                )
                ar_model = accelerator.prepare(ar_model)
                ar_model.train()
            else:
                accelerator.load_state(os.path.join(general_config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, general_config.num_training_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if ema_config.use_ema:
        if ema_config.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)

    # Now you train the model
    accelerator.wait_for_everyone()
    if args.lat_weighted_loss:
        loss_lat_weight = get_normalized_lat_weights_based_on_cos(
            np.linspace(-83.25, 84.75, 15)
        )  # (lat,)
        loss_lat_weight = torch.from_numpy(loss_lat_weight).to(
            accelerator.device
        )  # (lat,)
        loss_lat_weight = loss_lat_weight.view(1, 1, 1, -1, 1)  # (1, 1, 1, lat, 1)
    for epoch in range(first_epoch, general_config.num_train_epochs):
        ar_model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ar_model):
                # initial_profile: (B, C, T, H, W), clean_images: (B, C, T, H, W), timestamps: (B,)
                initial_profile, clean_images, timestamps = batch
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                # diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
                if not general_config.do_edm_style_training:
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bs,),
                        device=clean_images.device,
                    )
                    timesteps = timesteps.long()
                else:
                    # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    # The scheduler init and step has: self.timesteps = self.precondition_noise(sigmas)
                    # indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,))
                    indices = noise_sampler(
                        bs,
                        cur_step=global_step,
                        generator=noise_sampler_gen,
                        device="cpu",
                    )
                    # print(f"Process {accelerator.process_index}, step {step}, indices: {torch.mean(indices.float())}")
                    gathered_indices = accelerator.gather(
                        indices.to(accelerator.device)
                    )
                    # print(f"Process {accelerator.process_index}, step {step}, gathered_indices: {gathered_indices}")
                    accelerator.log(
                        {"mean_noise_level": torch.mean(gathered_indices.float())},
                        step=global_step,
                    )
                    timesteps = noise_scheduler.timesteps[indices].to(
                        device=clean_images.device
                    )

                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                if general_config.do_edm_style_training:
                    # sigmas: from high to low, default: 80 -> 0.002
                    sigmas = get_sigmas(
                        noise_scheduler,
                        timesteps,
                        len(noisy_images.shape),
                        noisy_images.dtype,
                        device=accelerator.device,
                    )
                    x_in = noise_scheduler.precondition_inputs(
                        noisy_images, sigmas
                    )  # scale_model_input designed for step

                model_pred = torch.fill(torch.empty_like(clean_images), float("nan"))
                for push_forward_step in range(args.num_push_forward_steps):
                    start_idx = push_forward_step * num_slice_per_push_forward
                    end_idx = (
                        push_forward_step + 1
                    ) * num_slice_per_push_forward  # excluded
                    tmp_x_in = x_in[:, :, start_idx:end_idx]
                    if push_forward_step >= 1:
                        # update, timestamps & initial_profile
                        for i in range(bs):
                            timestamps[i] = convert_datetime_to_int(
                                convert_int_to_datetime(timestamps[i].item())
                                + pd.Timedelta(hours=6)
                            )
                        if general_config.do_edm_style_training:
                            if "EDM" in noise_scheduler_config.target:
                                initial_profile = noise_scheduler.precondition_outputs(
                                    noisy_images[
                                        :,
                                        :,
                                        start_idx
                                        - train_dataloader_config.input_seq_len : start_idx,
                                    ],
                                    model_pred[
                                        :,
                                        :,
                                        start_idx
                                        - train_dataloader_config.input_seq_len : start_idx,
                                    ].detach(),
                                    sigmas,
                                )

                    model_pred[:, :, start_idx:end_idx] = ar_model(
                        tmp_x_in,
                        timesteps,
                        initial_profile,
                        time_elapsed=timestamps,
                        return_dict=False,
                    )[0]

                weighting = None
                if general_config.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if "EDM" in noise_scheduler_config.target:
                        model_pred = noise_scheduler.precondition_outputs(
                            noisy_images, model_pred, sigmas
                        )  # the last (or more) channel is the mask
                        weighting = (sigmas**2 + 0.5**2) / (
                            sigmas * 0.5
                        ) ** 2  # assume sigma_data=0.5 for now
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_images
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (
                                -sigmas / (sigmas**2 + 1) ** 0.5
                            ) + (noisy_images / (sigmas**2 + 1))
                    # (comment from diffuser) We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    # (my comment) The EDM weight -> faster convergence
                    if "EDM" not in noise_scheduler_config.target:
                        weighting = (sigmas**-2.0).float()
                    # loss = (weighting.float() * ((clean_images.float() - model_output.float()) ** 2)).mean()
                    # loss = ((clean_images.float() - model_output.float()) ** 2).mean()
                    # loss = loss_fn(model_output, clean_images, sigmas)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = (
                        clean_images if general_config.do_edm_style_training else noise
                    )
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        clean_images
                        if general_config.do_edm_style_training
                        else noise_scheduler.get_velocity(
                            clean_images, noise, timesteps
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if general_config.snr_gamma is None:
                    if weighting is not None:
                        if args.lat_weighted_loss:
                            loss = torch.mean(
                                (
                                    loss_lat_weight.float()
                                    * weighting.float()
                                    * (model_pred.float() - target.float()) ** 2
                                ),
                            )
                        else:
                            loss = torch.mean(
                                (
                                    weighting.float()
                                    * (model_pred.float() - target.float()) ** 2
                                ),
                            )
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float())
                else:
                    # TODO: udpate eps prediction
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack(
                            [
                                snr,
                                general_config.snr_gamma * torch.ones_like(timesteps),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                train_loss += loss.item() / accelerator.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(ar_model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if ema_config.use_ema:
                    if ema_config.offload_ema:
                        ema_model.to(device="cuda", non_blocking=True)
                    ema_model.step(ar_model.parameters())
                    if ema_config.offload_ema:
                        ema_model.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                logs = {
                    "train loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                if ema_config.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                global_step += 1
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % general_config.checkpointing_steps == 0:
                        # https://github.com/huggingface/accelerate/issues/314
                        # when using with accelerator compile, the first time could take a while and cause a timeout
                        logger.info(f"Saving checkpoint at step {global_step}")

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(general_config.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        general_config.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)
                if global_step % general_config.checkpointing_steps == 0:
                    save_path = os.path.join(
                        general_config.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if ema_config.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)

            if global_step >= general_config.num_training_steps:
                break

        if (
            (epoch + 1) % general_config.save_image_epochs == 0
            or epoch == general_config.num_train_epochs - 1
        ):
            logger.info(f"Logging validation for epoch {epoch}")
            if accelerator.is_main_process:
                if ema_config.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_model.store(ar_model.parameters())
                    ema_model.copy_to(ar_model.parameters())
            accelerator.wait_for_everyone()
            with accelerator.split_between_processes(
                val_timerange4pred
            ) as input_timestamp:
                # print(f'process {accelerator.process_index} is processing {input_timestamp}')
                log_validation(
                    "val",
                    xr_dataset=val_dataset,
                    config=general_config,
                    ar_model=unwrap_model(ar_model),
                    full_field_mean_tensor=full_field_mean_tensor,
                    full_field_std_tensor=full_field_std_tensor,
                    input_seq_len=train_dataloader_config.input_seq_len,
                    return_seq_len=num_slice_per_push_forward,
                    encdec_model=encdec_model,
                    latent_transform_func=train_dataloader.dataset.transform,
                    latent_inv_transform_func=train_dataloader.dataset.inv_transform,
                    noise_scheduler_config=noise_scheduler_config,
                    accelerator=accelerator,
                    timestamp_list=input_timestamp,
                    ensemble_size=10,
                    num_inference_steps=20,
                    eval_ms=False,
                )
            if accelerator.is_main_process:
                if ema_config.use_ema:
                    # Restore the UNet parameters.
                    ema_model.restore(ar_model.parameters())
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if (
                (epoch + 1) % general_config.save_model_epochs == 0
                or epoch == general_config.num_train_epochs - 1
            ):
                # save the model
                logger.info(f"Saving model at epoch {epoch + 1}")

                if ema_config.use_ema:
                    ema_model.store(ar_model.parameters())
                    ema_model.copy_to(ar_model.parameters())

                unwrap_model(ar_model).save_pretrained(
                    os.path.join(general_config.output_dir, "ar_model")
                )

                if ema_config.use_ema:
                    ema_model.restore(ar_model.parameters())

                if epoch == general_config.num_train_epochs - 1 and args.push_to_hub:
                    upload_folder(
                        repo_id=args.hub_model_id,
                        folder_path=general_config.output_dir + "/ar_model",
                        path_in_repo=general_config.output_dir.split("/")[-1],
                        commit_message="running weight",
                        ignore_patterns=["checkpoint_"],
                        token=args.hub_token if args.hub_token else None,
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
