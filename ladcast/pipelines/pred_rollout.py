import argparse
import json
import math
import os
import pprint
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from ladcast.dataloader.ar_dataloder import convert_datetime_to_int
from ladcast.dataloader.utils import (
    filter_time_range,
    get_inv_transform_3D,
    get_transform_3D,
    tensor_to_xarr,
)
from ladcast.models.DCAE import AutoencoderDC
from ladcast.models.LaDCast_3D_model import LaDCastTransformer3DModel
from ladcast.pipelines.pipeline_AR import AutoRegressive2DPipeline
from ladcast.pipelines.utils import roll_out_serial
from ladcast.utils import instantiate_from_config


logger = get_logger(__name__, log_level="INFO")

var_list = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "total_precipitation_6hr",
]


noise_scheduler_config = {
    "target": "diffusers.EDMDPMSolverMultistepScheduler",
    "param": {"sigma_data": 0.5},
}


def fix_time_encoding_for_zarr(ds):
    """
    Fix time encoding for zarr serialization to ensure correct time storage.
    Automatically determines the reference date from the dataset.
    """
    ds = ds.copy()
    min_time = pd.Timestamp(ds.time.min().values)
    reference_date = min_time.strftime("%Y-%m-%d")
    ds.time.encoding = {
        "dtype": "float64",
        "units": f"hours since {reference_date}",
        "calendar": "proleptic_gregorian",
        "_FillValue": None,
        "zlib": True,
    }

    return ds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a VAE model on a dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the separated variables zarr",
    )
    parser.add_argument(
        "--latent_normal_json",
        type=str,
        default=None,
        help="Path to the latent transform json",
    )
    parser.add_argument(
        "--encdec_model_name", type=str, default=None, help="HF hub model name"
    )
    parser.add_argument(
        "--encdec_model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--encdec_model_class",
        type=str,
        default="dcae",
        help="The encoder-decoder model class: dcvae or vae",
    )
    parser.add_argument(
        "--ar_model_name", type=str, default=None, help="HF hub model name"
    )
    parser.add_argument(
        "--ar_model_path", type=str, default=None, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2018-01-01",
        help="Start date for the evaluation",
    )
    parser.add_argument(
        "--end_date", type=str, default="2018-12-31", help="End date for the evaluation"
    )
    parser.add_argument(
        "--num_samples_per_month",
        type=int,
        default=None,
        help="Number of samples per month",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=15,
        help="Number of ensemble members to use",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument(
        "--total_lead_time_hour", type=int, default=240, help="Total lead time in hours"
    )
    parser.add_argument(
        "--dataset_interval_hour",
        type=int,
        default=1,
        help="Interval between each dataset",
    )
    parser.add_argument(
        "--log_pred_interval_hour",
        type=int,
        default=12,
        help="Interval between each prediction logged",
    )
    parser.add_argument(
        "--step_size_hour", type=int, default=6, help="Step size in hours"
    )
    parser.add_argument(
        "--input_seq_len", type=int, default=1, help="Input sequence length"
    )
    parser.add_argument(
        "--return_seq_len",
        type=int,
        default=4,
        help="Number of frames for each call of the pipeline",
    )
    parser.add_argument(
        "--normalization_json",
        type=str,
        default="ERA5_normal.json",
        help="Path to the normalization JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output zarr file or directory if save as latent",
    )
    parser.add_argument(
        "--load_ds_in_memory", action="store_true", help="Load the dataset in memory"
    )
    parser.add_argument(
        "--pressure_levels",
        type=int,
        nargs="+",
        default=[500, 700, 850],
        help="List of pressure levels to log (in hPa). Default: [500, 700, 850]",
    )
    parser.add_argument(
        "--sampler_type", type=str, default="edm", help="Sampling type: pipeline or edm"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lsm_path", type=str, default=None, help="Path to the land-sea mask tensor."
    )
    parser.add_argument(
        "--orography_path", type=str, default=None, help="Path to the orography tensor."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of time indices to process in each batch",
    )
    parser.add_argument(
        "--ar_cls",
        type=str,
        default="transformer",
        help="The Autoregressive model backbone.",
    )
    parser.add_argument(
        "--save_as_latent",
        action="store_true",
        help="Whether to save the output as latent space",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0,
        help="Noise level for perturbing the latent",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    with open(args.latent_normal_json) as f:
        transform_args = json.load(f)
        transform_args["target_std"] = 0.5
    transform = get_transform_3D("normalize", transform_args)
    inv_transform = get_inv_transform_3D("normalize", transform_args)

    with open(args.normalization_json) as f:
        normalization_param_dict = json.load(f)

    ref_start_time = pd.to_datetime(args.start_date) - pd.Timedelta(
        hours=args.step_size_hour * (args.input_seq_len - 1)
    )
    ref_end_time = pd.to_datetime(args.end_date) + pd.Timedelta(
        hours=args.total_lead_time_hour
    )
    ref_time_range = pd.date_range(
        start=ref_start_time, end=ref_end_time, freq=f"{args.step_size_hour}h"
    )
    # print(f"ref_time_range: {ref_time_range}")
    full_time_range = pd.date_range(
        start=args.start_date, end=args.end_date, freq=f"{args.log_pred_interval_hour}h"
    )
    if args.num_samples_per_month is not None:
        tmp_time_range = filter_time_range(
            full_time_range,
            num_samples_per_month=args.num_samples_per_month,
            enforce_year="2018",
        )
    # log_time_range = pd.DatetimeIndex([date for date in full_time_range if date not in tmp_time_range])
    log_time_range = tmp_time_range
    ds = xr.open_zarr(args.data_path).sel(time=ref_time_range)
    ds = ds.sel(latitude=slice(-88.5, 90))  # crop south pole

    accelerator = Accelerator()

    if accelerator.is_main_process:
        pprint.pp(args.__dict__)

    lsm_tensor = torch.from_numpy(
        ds["land_sea_mask"].transpose("latitude", "longitude").values
    ).float()
    if args.lsm_path:
        lsm_tensor = torch.load(args.lsm_path, weights_only=True)  # (lat, lon)
        lsm_tensor = lsm_tensor[1:, :]  # crop south pole (first row)
    if args.orography_path:
        # ['standard_deviation_of_orography', 'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography', 'slope_of_sub_gridscale_orography']
        orography_tensor = torch.load(
            args.orography_path, weights_only=True
        )  # (4, lat, lon)
        orography_tensor = orography_tensor[:, 1:, :]  # crop south pole (first row)

    static_conditioning_tensor = None
    if args.lsm_path is not None:
        static_conditioning_tensor = lsm_tensor.unsqueeze(0)  # (1, lat, lon)
        if args.orography_path is not None:
            static_conditioning_tensor = torch.cat(
                [static_conditioning_tensor, orography_tensor], dim=0
            )
    elif args.orography_path is not None:
        static_conditioning_tensor = orography_tensor

    # normalize static conditioning tensor if it exists
    if static_conditioning_tensor is not None:
        static_mean_tensor = static_conditioning_tensor.mean(
            dim=(1, 2), keepdim=True
        )  # (C, 1, 1)
        static_std_tensor = static_conditioning_tensor.std(dim=(1, 2), keepdim=True)
        static_conditioning_tensor = (
            static_conditioning_tensor - static_mean_tensor
        ) / static_std_tensor

    ds = ds[var_list]

    if args.load_ds_in_memory:
        ds = ds.load()

    repo_name = "tonyzyl/ladcast"
    if args.encdec_model_class == "dcae":
        encdec_model_cls = AutoencoderDC
    else:
        raise NotImplementedError(
            f"Unknown encoder-decoder model class: {args.encdec_model_class}"
        )
    if args.encdec_model_path is not None:
        encdec_model = encdec_model_cls.from_pretrained(args.encdec_model_path)
    elif args.encdec_model_name is not None:
        encdec_model = encdec_model_cls.from_pretrained(
            repo_name, subfolder=args.encdec_model_name
        )
    else:
        raise ValueError("Please provide a valid VAE model path or name")

    if args.ar_cls == "transformer":
        ar_model_cls = LaDCastTransformer3DModel
    else:
        raise NotImplementedError(f"Unknown autoregressive model class: {args.ar_cls}")

    if args.ar_model_path is not None:
        ar_model = ar_model_cls.from_pretrained(args.ar_model_path)
    elif args.ar_model_name is not None:
        ar_model = ar_model_cls.from_pretrained(repo_name, subfolder=args.ar_model_name)
    else:
        raise ValueError("Please provide a valid AR model path or name")

    assert args.output is not None, "Please provide a valid output path"

    ar_model = ar_model.eval()
    encdec_model = encdec_model.eval()
    ar_model = ar_model.to(accelerator.device)
    encdec_model = encdec_model.to(accelerator.device)

    noise_scheduler = instantiate_from_config(noise_scheduler_config)
    pipeline = AutoRegressive2DPipeline(ar_model, scheduler=noise_scheduler)

    # Determine batch size and number of batches
    batch_size = (
        args.batch_size if args.batch_size is not None else accelerator.num_processes
    )
    assert args.log_pred_interval_hour % args.dataset_interval_hour == 0, (
        "log_pred_interval_hour must be a multiple of dataset_interval_hour, got {args.log_pred_interval_hour} and {args.dataset_interval_hour}"
    )
    assert args.step_size_hour % args.dataset_interval_hour == 0, (
        "step_size_hour must be a multiple of dataset_interval_hour, got {args.step_size_hour} and {args.dataset_interval_hour}"
    )
    num_batches = math.ceil(log_time_range.size / batch_size)

    # Iterate over the dataset in batches
    for batch_num in tqdm(range(num_batches), desc="Processing Batches"):
        if accelerator.is_main_process:
            rollout_start_time = time.time()
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, log_time_range.size)
        indices_of_dataset_in_batch = list(range(start_idx, end_idx))

        # accelerate split&gather: https://github.com/huggingface/accelerate/issues/3393
        # https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py
        with accelerator.split_between_processes(
            indices_of_dataset_in_batch
        ) as input_indices:
            # Generate predictions for the current batch
            pred_timestamp = log_time_range[input_indices]
            # print("Process ", accelerator.process_index, " corresponding to time: ", pred_timestamp)
            logger.info(
                f"Process idx: {accelerator.process_index}, time: {pred_timestamp}"
            )
            result_tensor = roll_out_serial(
                xr_dataset=ds,
                pred_timestamp=pred_timestamp,
                pipeline=pipeline,
                ensemble_size=args.ensemble_size,
                return_seq_len=args.return_seq_len,
                num_inference_steps=args.num_inference_steps,
                normalization_param_dict=normalization_param_dict,
                encdec_model=encdec_model,
                encdec_model_type="ae",
                static_tensor4encdec=static_conditioning_tensor,
                latent_transform="normalize",
                latent_transform_args=transform_args,
                sampler_type=args.sampler_type,
                input_seq_len=args.input_seq_len,
                dataset_interval_hour=args.dataset_interval_hour,
                total_lead_time_hour=args.total_lead_time_hour,
                log_pred_interval_hour=args.log_pred_interval_hour,
                step_size_hour=args.step_size_hour,
                return_tensor=True,
                return_latent=args.save_as_latent,
                noise_level=args.noise_level,
            )  # (ensemble_size, C, T, H, W)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            rollout_end_time = time.time()
        if accelerator.is_main_process:
            # print("Process: ", accelerator.process_index, ", before gather result_tensor shape: ", result_tensor.shape)
            logger.info(
                f"Process {accelerator.process_index}, before gather result_tensor shape: {result_tensor.shape}"
            )
        gathered_tensor = accelerator.gather(
            result_tensor.to(accelerator.device)
        )  # (num_pred, ensemble_size, C, T, H, W)
        gathered_timestamps = accelerator.gather_for_metrics(
            pred_timestamp, use_gather_object=True
        )
        if accelerator.is_main_process:
            # print("Process: ", accelerator.process_index, ", after gather result_tensor shape: ", gathered_tensor.shape)
            logger.info(
                f"Process {accelerator.process_index}, after gather result_tensor shape: {gathered_tensor.shape}"
            )
        gathered_tensor = gathered_tensor[
            : len(indices_of_dataset_in_batch)
        ]  # drop duplicate tensors
        gathered_timestamps = gathered_timestamps[
            : len(indices_of_dataset_in_batch)
        ]  # drop duplicate timestamps
        if accelerator.is_main_process:
            # print("Process: ", accelerator.process_index, ", after dropping result_tensor shape: ", gathered_tensor.shape, ", timestamps: ", gathered_timestamps)
            logger.info(
                f"Process {accelerator.process_index}, after dropping result_tensor shape: {gathered_tensor.shape}, timestamps: {gathered_timestamps}"
            )
        if accelerator.is_main_process:
            if args.save_as_latent:
                for i, cur_timestamp in enumerate(gathered_timestamps):
                    tensor_np = gathered_tensor[i].cpu().numpy()
                    np.save(
                        os.path.join(
                            args.output,
                            f"latent_{convert_datetime_to_int(cur_timestamp)}.npy",
                        ),
                        tensor_np,
                    )
            else:
                coords = {
                    #'time': log_time_range[indices_of_dataset_in_batch],
                    "time": gathered_timestamps,
                    "level": ds.level.values,
                    "latitude": ds.latitude.values,
                    "longitude": ds.longitude.values,
                }
                result_xarr = xr.Dataset(coords=coords, data_vars=ds.data_vars)
                result_xarr = result_xarr.expand_dims(
                    {"idx": range(args.ensemble_size)}
                )
                result_xarr = (
                    result_xarr.expand_dims(
                        {
                            "prediction_timedelta": np.arange(
                                0,
                                (args.total_lead_time_hour // args.step_size_hour + 1),
                            )
                            * args.step_size_hour
                            * 3600
                            * 10**9
                        }
                    )
                    .copy(deep=True)
                    .transpose(
                        "idx",
                        "time",
                        "prediction_timedelta",
                        "level",
                        "latitude",
                        "longitude",
                    )
                )
                meta_coords = coords.copy()
                meta_coords["time"] = result_xarr.prediction_timedelta.values
                xarr_meta = xr.Dataset(coords=meta_coords, data_vars=ds.data_vars)

                # for cur_time_idx, current_time in enumerate(log_time_range[indices_of_dataset_in_batch]):
                for cur_time_idx, current_time in enumerate(gathered_timestamps):
                    for ens_idx in range(args.ensemble_size):
                        tmp_xarr = tensor_to_xarr(
                            gathered_tensor[cur_time_idx, ens_idx],
                            xarr_meta,
                            normalization_param_dict=normalization_param_dict,
                        )
                        tmp_xarr = tmp_xarr.rename({"time": "prediction_timedelta"})
                        for lead_time_idx, lead_time in enumerate(
                            result_xarr.prediction_timedelta.values
                        ):
                            single_time_slice = tmp_xarr.isel(
                                prediction_timedelta=lead_time_idx
                            )
                            # print('\n idx:', idx, 'time:', current_time, 'prediction_timedelta:', lead_time/3600/(10**9))
                            result_xarr.loc[
                                dict(
                                    idx=ens_idx,
                                    time=current_time,
                                    prediction_timedelta=lead_time,
                                )
                            ] = single_time_slice

                # Fix time encoding
                result_xarr = result_xarr.sel(level=args.pressure_levels)
                result_xarr = fix_time_encoding_for_zarr(result_xarr)

                # Append results to the output Zarr file
                if accelerator.is_main_process:
                    append_start_time = time.time()
                print(f"Processor {accelerator.process_index} writing to Zarr...")
                if os.path.exists(args.output):
                    result_xarr.to_zarr(
                        args.output, mode="a", append_dim="time", zarr_format=2
                    )
                else:
                    result_xarr.to_zarr(args.output, mode="w", zarr_format=2)
                if accelerator.is_main_process:
                    append_end_time = time.time()
                    print(
                        f"Rollout time {rollout_end_time - rollout_start_time:.4f} seconds, append time {append_end_time - append_start_time:.4f} seconds"
                    )

        accelerator.wait_for_everyone()

    print(f"Processing complete. Results saved to {args.output}")
