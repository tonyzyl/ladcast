import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
from accelerate import Accelerator
from accelerate.logging import get_logger

from ladcast.dataloader.utils import (
    precompute_mean_std,
    xarr_to_tensor,
    xarr_varname_to_tensor,
)
from ladcast.evaluate.utils import (
    climatology_to_timeseries,
    get_acc,
    get_normalized_lat_weights_based_on_cos,
    pointwise_crps_skill,
    pointwise_crps_spread,
)

# import wandb
from ladcast.models.DCAE import AutoencoderDC
from ladcast.models.embeddings import convert_int_to_datetime
from ladcast.pipelines.utils import decode_latent_ens

VARIABLE_NAMES = [
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

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
NUM_ATM_VARS = 6
NUM_SUR_VARS = 6
SST_CHANNEL_IDX = 82


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the latent forecast.")
    parser.add_argument(
        "--normalization_json",
        type=str,
        default="ERA5_normal.json",
        help="Path to the normalization JSON file",
    )
    parser.add_argument(
        "--encdec_model",
        type=str,
        default=None,
        help="Path to the encoder-decoder model on huggingface hub.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the reference ERA5 zarr dataset.",
    )
    parser.add_argument(
        "--result_path", type=str, default=None, help="Path to the result directory."
    )
    parser.add_argument(
        "--climatology_path",
        type=str,
        default=None,
        help="Path to the ERA5 climatology zarr dataset.",
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
        "--output", type=str, default=None, help="Path to the CSV file for logging."
    )
    parser.add_argument(
        "--step_size_hour", type=int, default=6, help="Step size in hours."
    )
    parser.add_argument(
        "--latent_spatial_scale",
        type=int,
        default=8,
        help="Spatial scale of the latent space.",
    )
    parser.add_argument(
        "--total_lead_time_hour",
        type=int,
        default=240,
        help="Total lead time in hours.",
    )
    parser.add_argument(
        "--load_ds_in_memory", action="store_true", help="Load the model in memory."
    )
    parser.add_argument(
        "--crop_init",
        action="store_true",
        help="Crop the ensemble initialization (t=0) if the latent contains it.",
    )
    parser.add_argument(
        "--force_ens_size",
        type=int,
        default=None,
        help="Force ensemble size for the evaluation.",
    )

    return parser.parse_args()


def main(args):
    logger = get_logger(__name__, log_level="INFO")

    if args.total_lead_time_hour % args.step_size_hour != 0:
        raise ValueError("total_lead_time_hour must be divisible by step_size_hour.")

    total_num_steps = int(args.total_lead_time_hour / args.step_size_hour)

    repo_name = "tonyzyl/ladcast"
    encdec_model = AutoencoderDC.from_pretrained(
        repo_name,
        subfolder=args.encdec_model,
    )

    accelerator = Accelerator(
        mixed_precision="no",
        # log_with="wandb",
    )

    encdec_model.to(accelerator.device)

    accelerator.init_trackers(
        project_name="evaluate_latent_forecast",
        config={
            "encdec_model_name": args.encdec_model_name,
            "data_path": args.data_path,
            "result_path": args.result_path,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "step_size_hour": args.step_size_hour,
            "latent_spatial_scale": args.latent_spatial_scale,
            "total_lead_time_hour": args.total_lead_time_hour,
            "load_ds_in_memory": args.load_ds_in_memory,
        },
    )

    logger.info("This message should appear from the main process")
    print("Using print() from main process")

    latitude = np.linspace(-88.5, 90, 120)  # crop south pole
    lat_weight = get_normalized_lat_weights_based_on_cos(latitude)  # (lat,)
    lat_weight = torch.from_numpy(lat_weight)  # (lat,)

    if args.normalization_json:
        with open(args.normalization_json, "r") as f:
            normalization_param_dict = json.load(f)
    else:
        raise ValueError("normalization_json is required.")

    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, VARIABLE_NAMES
    )

    ref_timerange = pd.date_range(
        start=args.start_date, end=args.end_date, freq=f"{args.step_size_hour}h"
    )
    ds = xr.open_zarr(args.data_path).sel(time=ref_timerange)
    ds = ds.sel(latitude=slice(-88.5, 90))  # crop south pole
    ds = ds[VARIABLE_NAMES]

    climate_ds = xr.open_zarr(args.climatology_path)
    climate_ds = climate_ds.sel(latitude=slice(-88.5, 90))  # crop south pole
    climate_ds = climate_ds[VARIABLE_NAMES]
    climate_ds = climate_ds.transpose(
        "dayofyear", "hour", "level", "latitude", "longitude"
    )
    """
    Dimensions:                  (hour: 4, dayofyear: 366, level: 13,
                              longitude: 240, latitude: 121)
    Coordinates:
    * dayofyear                (dayofyear) int64 3kB 1 2 3 4 5 ... 363 364 365 366
    * hour                     (hour) int64 32B 0 6 12 18
    * latitude                 (latitude) float64 968B -90.0 -88.5 ... 88.5 90.0
    * level                    (level) int64 104B 50 100 150 200 ... 850 925 1000
    * longitude                (longitude) float64 2kB 0.0 1.5 3.0 ... 357.0 358.5
    """

    if args.load_ds_in_memory:
        ds = ds.load()  # 1yr ~14.5G per process
        climate_ds = climate_ds.load()  # 1yr ~14.5G per process

    paths = sorted(glob.glob(os.path.join(args.result_path, "latent_*.npy")))
    tmp_time_str_list = [
        path.split("/")[-1].split("_")[-1].split(".")[0] for path in paths
    ]  # YYYYMMDDHH
    time_str_list = []
    # filter time_str_list, capped at end_date(str: e.g., 2018-01-01T00:00:00) - total_lead_time_hour(int)
    end_datetime = pd.to_datetime(args.end_date)
    max_forecast_datetime = end_datetime - pd.Timedelta(hours=args.total_lead_time_hour)

    for time_str in tmp_time_str_list:
        time_datetime = convert_int_to_datetime(int(time_str))
        if time_datetime <= max_forecast_datetime:
            time_str_list.append(time_str)

    encdec_model.eval()

    with torch.no_grad():
        with accelerator.split_between_processes(time_str_list) as process_str_list:
            timestamp_tensor = torch.full(
                (len(process_str_list),),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
            # single_mse = torch.full((len(process_str_list), 84, total_num_steps), float('nan'), dtype=torch.float32, device="cpu")
            ens_mse = torch.full(
                (len(process_str_list), 84, total_num_steps),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
            crps_spread = torch.full(
                (len(process_str_list), 84, total_num_steps),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
            crps_skill = torch.full(
                (len(process_str_list), 84, total_num_steps),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
            crps = torch.full(
                (len(process_str_list), 84, total_num_steps),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
            ens_acc = torch.full(
                (len(process_str_list), 84, total_num_steps),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
            # logger.info(f"Process: {accelerator.process_index}, assigned time_str_list: {process_str_list}")
            print(
                f"Process: {accelerator.process_index}, assigned time_str_list: {process_str_list}"
            )

            for idx, time_str in enumerate(process_str_list):
                # logger.info(f"Process: {accelerator.process_index}, processing time_str: {time_str}, remaining: {len(process_str_list) - idx - 1}")
                print(
                    f"Process: {accelerator.process_index}, processing time_str: {time_str}, remaining: {len(process_str_list) - idx - 1}"
                )
                latent = np.load(
                    os.path.join(args.result_path, f"latent_{time_str}.npy")
                )
                latent = torch.from_numpy(latent).to(
                    accelerator.device
                )  # (ens, C, T, H, W)
                if args.crop_init:
                    # crop the first time step
                    latent = latent[:, :, 1:, ...]  # (ens, C, T-1, H, W)
                if args.force_ens_size is not None:
                    latent = latent[: args.force_ens_size, ...]

                ens_size, C, T, h, w = latent.shape
                H = h * args.latent_spatial_scale
                W = w * args.latent_spatial_scale
                # 31.2 G each process for ens_size=50
                decoded_tensor = torch.full(
                    (ens_size, C, T, H, W),
                    float("nan"),
                    dtype=torch.float32,
                    device="cpu",
                )
                if accelerator.is_main_process:
                    timer_start = time.time()
                    print(
                        f"Process {accelerator.process_index}, start decoding latent tensor..."
                    )
                for i in range(ens_size):
                    decoded_tensor[i] = decode_latent_ens(
                        encdec_model, latent[i : i + 1], mean_tensor, std_tensor
                    )[0]
                if accelerator.is_main_process:
                    print(
                        f"Process {accelerator.process_index}, finished decoding tensor, elapsed time: {time.time() - timer_start:.2f} seconds"
                    )

                timestamp_int = int(time_str)
                timestamp_tensor[idx] = timestamp_int
                cur_datetime = convert_int_to_datetime(int(time_str))
                ref_timerange = pd.date_range(
                    start=cur_datetime + pd.Timedelta(hours=args.step_size_hour),
                    end=cur_datetime + pd.Timedelta(hours=args.total_lead_time_hour),
                    freq=f"{args.step_size_hour}h",
                )
                ref_tensor = xarr_to_tensor(
                    ds.sel(time=ref_timerange), VARIABLE_NAMES
                )  # ->(C, T, H, W), return unnormalized tensor

                tmp_climate_ds = climatology_to_timeseries(
                    climate_ds,
                    start_time=cur_datetime,
                    lead_time=args.total_lead_time_hour,
                    interval=args.step_size_hour,
                    exclude_start=True,
                )
                climate_tensor, _ = xarr_varname_to_tensor(
                    tmp_climate_ds, VARIABLE_NAMES
                )  # (C, T, H, W)

                if accelerator.is_main_process:
                    print(
                        f"Process {accelerator.process_index}, start calculating metric..."
                    )

                # decoded_tensor on CPU: (ens, C, T, H, W)
                # ref_tensor, climate_tensor on CPU: (C, T, H, W)
                for t in range(T):
                    # slice & move to GPU
                    dec_t = decoded_tensor[:, :, t].to(
                        accelerator.device
                    )  # (ens, C, H, W)
                    ref_t = ref_tensor[:, t].to(accelerator.device)  # (C, H, W)
                    clim_t = climate_tensor[:, t].to(accelerator.device)  # (C, H, W)
                    weights = lat_weight.view(1, -1, 1).to(
                        accelerator.device
                    )  # (1,H,W)

                    # 1) ACC
                    mean_t = dec_t.mean(dim=0)  # (C, H, W)
                    acc_t = get_acc(mean_t, ref_t, clim_t, weights)  # (C,)
                    ens_acc[idx, :, t] = acc_t.cpu()

                    # 2) Ensemble MSE
                    se_t = (mean_t - ref_t) ** 2 * weights  # (C, H, W)
                    # split channels
                    ens_mse[idx, :SST_CHANNEL_IDX, t] = (
                        se_t[:SST_CHANNEL_IDX].mean(dim=(1, 2)).cpu()
                    )
                    ens_mse[idx, SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1, t] = (
                        torch.nanmean(
                            se_t[SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1], dim=(1, 2)
                        )
                    ).cpu()
                    ens_mse[idx, SST_CHANNEL_IDX + 1 :, t] = (
                        se_t[SST_CHANNEL_IDX + 1 :].mean(dim=(1, 2)).cpu()
                    )

                    # 3) CRPS‐spread & skill & total
                    spread_t = (
                        pointwise_crps_spread(dec_t, ensemble_dim=0) * weights
                    )  # (C, H, W)
                    skill_t = (
                        pointwise_crps_skill(dec_t, ref_t.unsqueeze(0), 0) * weights
                    )
                    crps_t = skill_t - 0.5 * spread_t

                    # split channels for each
                    # spread
                    crps_spread[idx, :SST_CHANNEL_IDX, t] = (
                        spread_t[:SST_CHANNEL_IDX].mean(dim=(1, 2)).cpu()
                    )
                    crps_spread[idx, SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1, t] = (
                        torch.nanmean(
                            spread_t[SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1], dim=(1, 2)
                        )
                    ).cpu()
                    crps_spread[idx, SST_CHANNEL_IDX + 1 :, t] = (
                        spread_t[SST_CHANNEL_IDX + 1 :].mean(dim=(1, 2)).cpu()
                    )
                    # skill
                    crps_skill[idx, :SST_CHANNEL_IDX, t] = (
                        skill_t[:SST_CHANNEL_IDX].mean(dim=(1, 2)).cpu()
                    )
                    crps_skill[idx, SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1, t] = (
                        torch.nanmean(
                            skill_t[SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1], dim=(1, 2)
                        )
                    ).cpu()
                    crps_skill[idx, SST_CHANNEL_IDX + 1 :, t] = (
                        skill_t[SST_CHANNEL_IDX + 1 :].mean(dim=(1, 2)).cpu()
                    )
                    # total CRPS
                    crps[idx, :SST_CHANNEL_IDX, t] = (
                        crps_t[:SST_CHANNEL_IDX].mean(dim=(1, 2)).cpu()
                    )
                    crps[idx, SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1, t] = (
                        torch.nanmean(
                            crps_t[SST_CHANNEL_IDX : SST_CHANNEL_IDX + 1], dim=(1, 2)
                        )
                    ).cpu()
                    crps[idx, SST_CHANNEL_IDX + 1 :, t] = (
                        crps_t[SST_CHANNEL_IDX + 1 :].mean(dim=(1, 2)).cpu()
                    )

                    # cleanup
                    del dec_t, ref_t, clim_t, mean_t, se_t, spread_t, skill_t, crps_t
                    torch.cuda.empty_cache()

                if accelerator.is_main_process:
                    print(
                        f"Process {accelerator.process_index}, finished calculating metrics for time_str: {time_str}, elapsed time: {time.time() - timer_start:.2f} seconds"
                    )

                # np.save(os.path.join(args.output, f"{time_str}_single_mse.npy"), single_mse[idx].cpu().numpy())
                np.save(
                    os.path.join(args.output, f"{time_str}_ens_acc.npy"),
                    ens_acc[idx].cpu().numpy(),
                )
                np.save(
                    os.path.join(args.output, f"{time_str}_ens_mse.npy"),
                    ens_mse[idx].cpu().numpy(),
                )
                np.save(
                    os.path.join(args.output, f"{time_str}_crps_spread.npy"),
                    crps_spread[idx].cpu().numpy(),
                )
                np.save(
                    os.path.join(args.output, f"{time_str}_crps_skill.npy"),
                    crps_skill[idx].cpu().numpy(),
                )
                np.save(
                    os.path.join(args.output, f"{time_str}_crps.npy"),
                    crps[idx].cpu().numpy(),
                )

        accelerator.wait_for_everyone()
        timestamp_tensor = timestamp_tensor.to(accelerator.device)
        ens_acc = ens_acc.to(accelerator.device)
        # single_mse = single_mse.to(accelerator.device)
        ens_mse = ens_mse.to(accelerator.device)
        crps_spread = crps_spread.to(accelerator.device)
        crps_skill = crps_skill.to(accelerator.device)
        crps = crps.to(accelerator.device)

        if accelerator.is_main_process:
            # logger.info(f"Process {accelerator.process_index}, before gather result shape: {ens_mse.shape}")
            print(
                f"Process {accelerator.process_index}, before gather result shape: {ens_mse.shape}"
            )
        timestamp_tensor = accelerator.gather(timestamp_tensor)
        ens_acc = accelerator.gather(ens_acc)
        # single_mse = accelerator.gather(single_mse)
        ens_mse = accelerator.gather(ens_mse)
        crps_spread = accelerator.gather(crps_spread)
        crps_skill = accelerator.gather(crps_skill)
        crps = accelerator.gather(crps)
        if accelerator.is_main_process:
            logger.info(
                f"Process {accelerator.process_index}, after gather result shape: {ens_mse.shape}"
            )
            print(
                f"Process {accelerator.process_index}, after gather result shape: {ens_mse.shape}"
            )
            timestamp_tensor = timestamp_tensor.cpu().numpy()
            ens_acc = ens_acc.cpu().numpy()
            # single_mse = single_mse.cpu().numpy()
            ens_mse = ens_mse.cpu().numpy()
            crps_spread = crps_spread.cpu().numpy()
            crps_skill = crps_skill.cpu().numpy()
            crps = crps.cpu().numpy()

            # Save the results
            np.save(os.path.join(args.output, "timestamp.npy"), timestamp_tensor)
            np.save(os.path.join(args.output, "ens_acc.npy"), ens_acc)
            # np.save(os.path.join(args.output, "single_mse.npy"), single_mse)
            np.save(os.path.join(args.output, "ens_mse.npy"), ens_mse)
            np.save(os.path.join(args.output, "crps_spread.npy"), crps_spread)
            np.save(os.path.join(args.output, "crps_skill.npy"), crps_skill)
            np.save(os.path.join(args.output, "crps.npy"), crps)

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = parse_args()
    main(args)
