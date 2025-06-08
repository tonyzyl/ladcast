import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.utils
import xarray as xr
from diffusers.utils import BaseOutput
from einops import rearrange, repeat

from ladcast.dataloader.ar_dataloder import convert_datetime_to_int
from ladcast.dataloader.utils import (
    get_inv_transform_3D,
    get_transform_3D,
    inverse_normalize_transform_3D,
    precompute_mean_std,
    tensor_to_xarr,
    xarr_to_tensor,
)
from ladcast.pipelines.edm_sampler import edm_AR_sampler


@dataclass
class Fields2DPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        fields torch or numpy array of shape (batch_size, channels, height, width):
    """

    fields: Union[torch.tensor, np.ndarray]


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"):
    # modified from diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


@torch.no_grad()
def decode_latent_ens(
    encdec_model,
    latents: torch.Tensor,
    mean_tensor: Optional[torch.Tensor] = None,
    std_tensor: Optional[torch.Tensor] = None,
    extract_first: Optional[int] = None,
) -> torch.Tensor:
    """
    latents: (B, C, T, H, W)
    """
    B, _, T, _, _ = latents.shape
    if extract_first is None:
        extract_first = T
    return_latents = encdec_model.decode(
        rearrange(
            latents[:, :, :extract_first].to(encdec_model.device),
            "B C T H W -> (B T) C H W",
        )
    ).sample
    return_latents = rearrange(
        return_latents, "(B T) C H W -> B C T H W", B=B, T=extract_first
    )
    if mean_tensor is not None:
        return_latents = inverse_normalize_transform_3D(
            return_latents,
            mean_tensor.to(return_latents.device),
            std_tensor.to(return_latents.device),
        )
    return return_latents


@torch.no_grad()
def latent_ens_to_xarr(
    latents: Union[torch.Tensor, str],
    encdec_model: torch.nn.Module,
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    variable_names: List[str],  # len = num_atm_vars + num_sur_vars
    timestamp: Optional[str] = None,  # YYYYMMDDHH
    levels: Optional[List[int]] = [
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        850,
        925,
        1000,
    ],  # pressure levels
    extract_variables: Optional[List[str]] = None,  # variables to extract
    extract_ens_member_idx: Optional[List[int]] = None,  # ensemble members to extract
    extract_first: Optional[int] = None,  # extract the first N timesteps
    num_atm_vars: int = 6,
    num_sur_vars: int = 6,
    latent_spatial_scale: int = 8,
    step_size_hour: int = 6,
    lat_start_deg: float = -88.5,
    lat_end_deg: float = 90.0,
    lon_start_deg: float = 0.0,
    lon_end_deg: float = 358.5,
    interval_deg: float = 1.5,
) -> xr.Dataset:
    """
    timestamp: str, YYYYMMDDHH format, e.g., '2023010100'
    Decode latent ensemble, then pack into an xarray.Dataset
    shaped (idx, time, prediction_timedelta, [level,] latitude, longitude).
    """
    # sanity checks
    if variable_names is None or len(variable_names) != num_atm_vars + num_sur_vars:
        raise ValueError("variable_names must be length num_atm_vars + num_sur_vars")

    # load / decode latents → decoded_tensor(ens, C, T, H, W) ---
    if isinstance(latents, str):
        arr = np.load(latents)  # shape (1, ens, C, T, h, w) or (ens, C, T, h, w)
        if arr.ndim == 6:
            arr = arr[0]
            Warning(
                f"latents shape {arr.shape}, expecting (ens, C, T, h, w), using the first member"
            )
        timestamp = latents.split("/")[-1].split("_")[-1].split(".")[0]
        latents = torch.from_numpy(arr).to(encdec_model.device)
    else:
        assert timestamp is not None, "When passing a tensor, you must give a timestamp"

    # decode each member into the full-resolution grid
    ens_size, C, T, h, w = latents.shape
    # H = h * latent_spatial_scale
    # W = w * latent_spatial_scale

    # total_channels = num_atm_vars * len(levels) + num_sur_vars
    atm_var_names = variable_names[:num_atm_vars]
    sur_var_names = variable_names[num_atm_vars : num_atm_vars + num_sur_vars]

    if extract_variables is None:
        extract_variables = variable_names
    if extract_ens_member_idx is None:
        extract_ens_member_idx = list(range(ens_size))
    if extract_first is None:
        extract_first = T

    # time
    dt0 = pd.to_datetime(timestamp, format="%Y%m%d%H")

    coords = {
        "idx": extract_ens_member_idx,
        # "prediction_timedelta": np.arange(extract_first) * step_size_hour * 3600 * 10**9,
        "prediction_timedelta": [
            pd.Timedelta(hours=step_size_hour * i) for i in range(extract_first)
        ],
        "level": levels,
        "latitude": np.arange(lat_start_deg, lat_end_deg + 1e-6, interval_deg),
        "longitude": np.arange(lon_start_deg, lon_end_deg + 1e-6, interval_deg),
    }

    ds = xr.Dataset(coords=coords)

    output_channels = 0
    for var in extract_variables:
        output_channels += len(levels) if var in atm_var_names else 1

    # before the loop: build an empty array for each var
    for var in extract_variables:
        if var in atm_var_names:
            shape = (
                len(extract_ens_member_idx),
                extract_first,
                len(levels),
                len(np.arange(lat_start_deg, lat_end_deg + interval_deg, interval_deg)),
                len(np.arange(lon_start_deg, lon_end_deg + interval_deg, interval_deg)),
            )
            ds[var] = xr.DataArray(
                np.full(shape, np.nan, dtype=np.float32),
                dims=("idx", "prediction_timedelta", "level", "latitude", "longitude"),
                coords=coords,
            )
        else:  # surface vars
            shape = (
                len(extract_ens_member_idx),
                extract_first,
                len(np.arange(lat_start_deg, lat_end_deg + interval_deg, interval_deg)),
                len(np.arange(lon_start_deg, lon_end_deg + interval_deg, interval_deg)),
            )
            ds[var] = xr.DataArray(
                np.full(shape, np.nan, dtype=np.float32),
                dims=("idx", "prediction_timedelta", "latitude", "longitude"),
                coords={
                    k: coords[k]
                    for k in ("idx", "prediction_timedelta", "latitude", "longitude")
                },
            )

    # decoded_tensor = torch.full((len(extract_ens_member_idx), output_channels, extract_first, H, W), float('nan'), dtype=torch.float32, device="cpu")
    for i, ens_idx in enumerate(extract_ens_member_idx):
        out = decode_latent_ens(
            encdec_model,
            latents[ens_idx : ens_idx + 1],
            mean_tensor=mean_tensor,
            std_tensor=std_tensor,
            extract_first=extract_first,
        )[0]  # returns shape (C, extract_first, H, W)
        for var in extract_variables:
            if var in atm_var_names:
                var_idx = atm_var_names.index(var)
                start = var_idx * len(levels)
                end = start + len(levels)
                block = out[start:end, ...]  # (lev, T, H, W)
                # reorder to (T, lev, H, W) then assign
                ds[var].values[i, :, :, :, :] = block.permute(1, 0, 2, 3).cpu().numpy()
            else:
                var_idx = sur_var_names.index(var)
                start = num_atm_vars * len(levels) + var_idx
                block = out[start, ...]  # (T, H, W)
                ds[var].values[i, :, :, :] = block.cpu().numpy()

    ds = ds.expand_dims({"time": [dt0]}).transpose(
        "idx", "time", "prediction_timedelta", "level", "latitude", "longitude"
    )

    ds = ds.chunk(
        {
            "idx": 1,
            "time": 1,
            "prediction_timedelta": T,
            "level": len(coords["level"]),
            "latitude": coords["latitude"].size,
            "longitude": coords["longitude"].size,
        }
    )

    return ds


@torch.no_grad()
def roll_out_serial(
    xr_dataset: xr.Dataset,
    pred_timestamp: pd.DatetimeIndex,
    pipeline,
    normalization_param_dict: Dict,
    ensemble_size: int = 1,
    num_inference_steps: int = 20,
    return_seq_len: int = 8,
    return_ensemble_mean: bool = False,
    encdec_model: torch.nn.Module = None,
    encdec_model_type: str = "vae",
    static_tensor4encdec: Optional[torch.Tensor] = None,  # [C, H, W]
    latent_transform: Optional[str] = "normalize",
    latent_transform_args: Optional[Dict] = None,
    total_lead_time_hour: int = 240,
    step_size_hour: int = 6,
    dataset_interval_hour: int = 1,
    sampler_type: Optional[str] = "pipeline",  # 'pipeline' or 'edm'
    generator=None,
    input_seq_len: int = 1,  # New parameter for sequence length
    return_tensor: bool = False,  # Whether to return a tensor instead of an xarray.Dataset
    return_latent: bool = False,  # Whether to return the latent tensor instead of the decoded tensor
    noise_level: Optional[float] = 0,  # Noise level for the latent space
) -> Union[xr.Dataset, torch.Tensor]:
    """
    Generate predictions using the AutoRegressive2DPipeline for sequence-to-sequence and store them in an xarray.Dataset.
    Note the input_torch_dataset shall acount for the time delay of the input sequence, e.g., (t-6h, t)
    Note that if return_latent is True, the t=0 is the compressed latent, not the ground truth.

    Args:
        input_torch_dataset (torch.utils.data.Dataset): The input dataset containing the known latents.
            The .ds attribute should contain the xarray.Dataset.
        pipeline: The prediction pipeline to use.
        prediction_config (Dict): Configuration dictionary for prediction.
        encdec_model (Optional[torch.nn.Module]): Optional VAE model for encoding/decoding latents.
        is_encdec_model_deterministic (bool): Whether the VAE is deterministic.
        latent_transform (Optional[str]): Transformation to apply to latents.
        latent_transform_args (Optional[Dict]): Arguments for the latent transformation.
        total_lead_time_hour (int): Total lead time in hours for predictions.
        step_size_hour (int): Step size in hours between predictions.
        sampler_type (Optional[str]): Type of sampler to use ('pipeline' or 'edm').
        generator: Random generator for reproducibility.
        input_seq_len (int): Length of the input sequence for sequence-to-sequence predictions.

    Returns:
        xr.Dataset: The output dataset containing the generated predictions.
        if return_tensor is True, return a tensor of shape (Num_pred, ensemble_size, C, T_timedelta, H, W)
    """

    if not return_ensemble_mean:
        return_size = ensemble_size
    else:
        return_size = 1
    # import time

    if total_lead_time_hour % step_size_hour != 0:
        raise ValueError("total_lead_time_hour must be divisible by step_size_hour.")
    total_num_steps = int(total_lead_time_hour / step_size_hour)
    num_repetitions = math.ceil(total_num_steps / return_seq_len)

    latent_transform_func = get_transform_3D(latent_transform, latent_transform_args)
    latent_inv_transform_func = get_inv_transform_3D(
        latent_transform, latent_transform_args
    )

    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, xr_dataset.data_vars
    )

    # assert input_torch_dataset.data_augmentation == False, "Data augmentation should be disabled for roll_out."
    assert step_size_hour % dataset_interval_hour == 0, (
        "step_size_hour must be divisible by dataset_interval_hour."
    )
    if return_latent:
        if return_ensemble_mean:
            raise ValueError(
                "return_ensemble_mean must be False when return_latent is True."
            )

    # start_date = pd.to_datetime(xr_dataset['time'].values[(input_seq_len-1)*(step_size_hour//dataset_interval_hour)])
    # end_date = pd.to_datetime(xr_dataset['time'].values[-1])
    # time_range = pd.date_range(start=start_date, end=end_date, freq=f'{log_pred_interval_hour}h')
    # print(f"Start date: {start_date}, End date: {end_date}, Time range: {time_range}")
    if not return_tensor:
        if return_latent:
            coords = {
                "idx": np.arange(ensemble_size),
                "time": pred_timestamp,
                "C": np.arange(encdec_model.config.latent_channels),
                "prediction_timedelta": np.arange(0, (total_num_steps + 1))
                * step_size_hour
                * 3600
                * 10**9,
                "H": np.arange(15),
                "W": np.arange(30),
            }
            tmp_shape = tuple(
                len(coords[d])
                for d in ("idx", "time", "C", "prediction_timedelta", "H", "W")
            )
            output_dataset = xr.Dataset(
                {
                    "latents": (
                        ["idx", "time", "C", "prediction_timedelta", "H", "W"],
                        np.full(tmp_shape, float("nan"), dtype=np.float32),
                    )
                },
                coords=coords,
            )
        else:
            coords = {
                #'idx': np.arange(return_size),
                #'time': time_range,
                "time": pred_timestamp,
                "latitude": xr_dataset["latitude"].values,
                "longitude": xr_dataset["longitude"].values,
            }
            if "level" in xr_dataset:
                coords["level"] = xr_dataset["level"].values
            else:
                print("No 'level' dim provided, use the default 13 levels.")
                coords["level"] = [
                    50,
                    100,
                    150,
                    200,
                    250,
                    300,
                    400,
                    500,
                    600,
                    700,
                    850,
                    925,
                    1000,
                ]
            data_vars = (
                xr_dataset.data_vars
            )  # get all variables in form of dict: {'var_name': (dims), ...}
            output_dataset = xr.Dataset(data_vars=data_vars, coords=coords)
            output_dataset = output_dataset.expand_dims({"idx": np.arange(return_size)})
            output_dataset = (
                output_dataset.expand_dims(
                    {
                        "prediction_timedelta": np.arange(0, (total_num_steps + 1))
                        * step_size_hour
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
            # print("output_dataset:", output_dataset)
    else:
        if return_latent:
            output_tensor = torch.full(
                (
                    len(pred_timestamp),
                    return_size,
                    encdec_model.config.latent_channels,
                    total_num_steps + 1,
                    15,
                    30,
                ),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )
        else:
            output_tensor = torch.full(
                (
                    len(pred_timestamp),
                    return_size,
                    encdec_model.config.out_channels
                    - encdec_model.config.static_channels,
                    total_num_steps + 1,
                    len(xr_dataset.latitude),
                    len(xr_dataset.longitude),
                ),
                float("nan"),
                dtype=torch.float32,
                device="cpu",
            )  # (len_pred, ensemble_size, C, T, H, W)

    # Iterate over each time step in the input dataset, starting from the idx of input_seq_len (the t=0)
    # for time_idx in range((input_seq_len-1)*(step_size_hour//dataset_interval_hour), len(xr_dataset['time'].values), (log_pred_interval_hour//dataset_interval_hour)):
    # current_time = xr_dataset['time'].values[time_idx]
    for cur_pred_idx, current_time in enumerate(pred_timestamp):
        # Extract the known_latents for the current input sequence
        # if input_seq_len > 1:
        # input_indices = range(time_idx - (input_seq_len-1)*(step_size_hour//dataset_interval_hour), time_idx + 1, (step_size_hour//dataset_interval_hour))
        # else:
        # input_indices = [time_idx]
        input_datetime_list = [
            current_time - pd.Timedelta(hours=step_size_hour * i)
            for i in range(input_seq_len - 1, -1, -1)
        ]
        # print('input_datetime_indices:', input_datetime_list)
        # known_latents = torch.stack([input_torch_dataset[idx] for idx in input_indices], dim=0)  # (input_seq_len, C, H, W)
        input_tensor = xarr_to_tensor(
            xr_dataset.sel(time=input_datetime_list),
            mean_tensor=mean_tensor,
            std_tensor=std_tensor,
        )  # (C, T, H, W)
        if return_tensor and not return_latent:
            # output_tensor[cur_pred_idx, :, :, 0, :, :] = input_tensor[:, -1, :, :].clone().expand(return_size, -1, -1, -1)
            output_tensor[cur_pred_idx, :, :, 0, :, :] = (
                xarr_to_tensor(xr_dataset.sel(time=[input_datetime_list[-1]]))[:, -1]
                .unsqueeze(0)
                .expand(return_size, -1, -1, -1)
            )

        # Encode the known latents using the VAE
        encoded_latents = encdec_model.encode(
            input_tensor.permute(1, 0, 2, 3).to(encdec_model.device),
            static_conditioning_tensor=static_tensor4encdec.unsqueeze(0).to(
                encdec_model.device
            ),
        )
        if encdec_model_type == "ae":
            known_latents = encoded_latents.latent  # (T, C, H, W)
        else:
            raise ValueError("Unknown encdec_model_type.")

        known_latents = known_latents.permute(
            1, 0, 2, 3
        )  # -> (C, input_seq_len(T), H, W)

        if return_tensor and return_latent:
            # (len_pred, ensemble_size, C, T, H, W)
            output_tensor[cur_pred_idx, :, :, 0, :, :] = (
                known_latents.clone()[:, -1]
                .unsqueeze(0)
                .expand(return_size, -1, -1, -1)
            )

        # Store the last input sequence with prediction_timedelta=0
        if not return_tensor:
            if return_latent:
                output_dataset["latents"].loc[
                    dict(
                        idx=np.arange(ensemble_size),
                        time=current_time,
                        prediction_timedelta=0,
                    )
                ] = (
                    known_latents.clone()[:, -1].unsqueeze(0).to("cpu").numpy()
                )  # -> (1, C, H, W)
            else:
                xarr_meta = xr_dataset.sel(time=current_time).transpose(
                    "level", "latitude", "longitude"
                )
                output_dataset.loc[dict(time=current_time, prediction_timedelta=0)] = (
                    xarr_meta.compute()
                )

        known_latents = latent_transform_func(
            known_latents
        )  # Apply any necessary transformations

        if noise_level > 0:
            latent_std_tensor = torch.tensor(
                latent_transform_args["std"], dtype=torch.float32
            ).to(known_latents.device)
            latent_std_tensor = latent_std_tensor[:, None, None, None]  # (C, 1, 1, 1)
            noise = (
                torch.randn_like(known_latents, device=known_latents.device)
                * noise_level
                * latent_std_tensor
            )
            known_latents = known_latents + noise

        known_latents = rearrange(known_latents, "C T H W -> 1 C T H W")

        # start_time = time.time()
        for step in range(num_repetitions):
            # Calculate the lead time for the current prediction step
            cur_step = min((1 + (step + 1) * return_seq_len), total_num_steps + 1)
            pred_selection = cur_step - (1 + step * return_seq_len)
            # return (ensemble_size, C, T, H, W)
            timestamps = current_time + pd.Timedelta(
                hours=step * step_size_hour * return_seq_len
            )
            timestamps = convert_datetime_to_int(timestamps)  # -> int format YYYYMMDDHH
            timestamps = torch.tensor(
                [timestamps], device=pipeline._execution_device
            )  # (1,)
            print("step:", step, "timestamps:", timestamps)
            sample_images = ensemble_AR_sampler(
                pipeline,
                sample_size=ensemble_size,
                return_seq_len=return_seq_len,
                num_inference_steps=num_inference_steps,
                known_latents=known_latents,
                timestamps=timestamps,
                sampler_type=sampler_type,
                device=pipeline._execution_device,
            )

            if not return_tensor:
                current_lead_time_list = output_dataset["prediction_timedelta"].values[
                    1 + step * return_seq_len : cur_step
                ]
            # Update known_latents for the next prediction step

            known_latents = sample_images[:, :, -input_seq_len:].clone()

            # (ensemble_size, C, T, H, W)
            sample_images = latent_inv_transform_func(
                rearrange(sample_images, "B C T H W -> C (B T) H W")
            )
            sample_images = rearrange(
                sample_images,
                "C (B T) H W -> B C T H W",
                B=ensemble_size,
                T=return_seq_len,
            )
            if not return_latent:
                # for ensemble_idx in range(ensemble_size):
                # sample_images[ensemble_idx] = latent_inv_transform_func(sample_images[ensemble_idx])  # Rescale latent
                # (T, C, H, W), treat T as batch dim then permute back to (C, T, H, W)
                # decoded_tensor[ensemble_idx] = encdec_model.decode(sample_images[ensemble_idx, :, :pred_selection].permute(1, 0, 2, 3)).sample.permute(1, 0, 2, 3).to('cpu')
                decoded_tensor = decode_latent_ens(
                    encdec_model,
                    sample_images[:, :, :pred_selection],
                    mean_tensor=mean_tensor,
                    std_tensor=std_tensor,
                )

            if not return_tensor:
                if return_latent:
                    output_dataset["latents"].loc[
                        dict(
                            idx=np.arange(ensemble_size),
                            time=current_time,
                            prediction_timedelta=current_lead_time_list,
                        )
                    ] = sample_images[:, :, :pred_selection].clone().to("cpu").numpy()
                else:
                    for idx in range(return_size):
                        coords = {
                            "idx": idx,
                            "time": current_lead_time_list,
                            "level": xr_dataset["level"].values,
                            "latitude": xr_dataset["latitude"].values,
                            "longitude": xr_dataset["longitude"].values,
                        }
                        xarr_meta = xr.Dataset(
                            coords=coords, data_vars=xr_dataset.data_vars
                        )  # empty xarray with specified coords and data vars' names & dimenmsion
                        if not return_ensemble_mean:
                            tmp_xarr = tensor_to_xarr(decoded_tensor[idx], xarr_meta)
                        else:
                            assert idx == 0, (
                                "when return_ensemble_mean is True, idx must be 0"
                            )
                            tmp_xarr = tensor_to_xarr(
                                decoded_tensor.mean(dim=0), xarr_meta
                            )
                        tmp_xarr = tmp_xarr.rename({"time": "prediction_timedelta"})

                        for i, lead_time in enumerate(current_lead_time_list):
                            single_time_slice = tmp_xarr.isel(prediction_timedelta=i)
                            # print('\n idx:', idx, 'time:', current_time, 'prediction_timedelta:', lead_time/3600/(10**9))
                            output_dataset.loc[
                                dict(
                                    idx=idx,
                                    time=current_time,
                                    prediction_timedelta=lead_time,
                                )
                            ] = single_time_slice
            else:
                if not return_ensemble_mean:
                    if return_latent:
                        output_tensor[
                            cur_pred_idx,
                            :,
                            :,
                            (1 + step * return_seq_len) : cur_step,
                            :,
                            :,
                        ] = (
                            sample_images[:, :, :pred_selection].clone().to("cpu")
                        )  # (ensemble_size, C, T, H, W)
                    else:
                        output_tensor[
                            cur_pred_idx,
                            :,
                            :,
                            (1 + step * return_seq_len) : cur_step,
                            :,
                            :,
                        ] = decoded_tensor.clone()  # (ensemble_size, C, T, H, W)
                else:
                    output_tensor[
                        cur_pred_idx, 0, :, (1 + step * return_seq_len) : cur_step, :, :
                    ] = decoded_tensor.mean(dim=0).clone()
        # end_time = time.time()
        # print(f"Time taken for one time step: {end_time - start_time:.2f} seconds")

    if return_tensor:
        return output_tensor
    else:
        return output_dataset


@torch.no_grad()
def ensemble_AR_sampler(
    pipeline,
    sample_size: int,
    return_seq_len: int,
    num_inference_steps: int,
    sampler_kwargs=None,
    known_latents: torch.Tensor = None,
    timestamps: Optional[torch.LongTensor] = None,  # int format YYYYMMDDHH
    batch_size: int = 64,
    sampler_type: Optional[str] = "edm",  # 'edm' or 'pipeline'
    device="cpu",
):
    """
    timestamp: (1,) or (B,), int format YYYYMMDDHH
    known_latents: (B, C, T, H, W)
    return shape (sample_size, C, T, H, W)
    """
    batch_size_list = [batch_size] * int(sample_size / batch_size) + [
        sample_size % batch_size
    ]
    # print(latents.shape, class_labels.shape, mask.shape, known_latents.shape)
    count = 0
    latents_shape = list(known_latents.shape)
    samples = torch.empty(
        sample_size,
        pipeline.ar_model.config.out_channels,
        return_seq_len,
        *latents_shape[-2:],
        device=device,
        dtype=pipeline.ar_model.dtype,
    )
    if sampler_kwargs is None:
        sampler_kwargs = {}
    if sampler_type == "edm":
        model = pipeline.ar_model
        noise_scheduler = copy.deepcopy(pipeline.scheduler)
    for num_sample in batch_size_list:
        # tmp_class_labels = repeat(class_labels, 'C -> B C', B=num_sample)
        generator = [
            torch.Generator("cpu").manual_seed(int(seed) % (1 << 32))
            for seed in range(count, count + num_sample)
        ]
        if known_latents.shape[0] == 1:
            # same initial condition
            tmp_known_latents = repeat(
                known_latents, "1 C T H W -> B C T H W", B=num_sample
            )
        else:
            tmp_known_latents = known_latents

        if sampler_type == "edm":
            tmp_samples = edm_AR_sampler(
                model,
                noise_scheduler,
                batch_size=num_sample,
                return_seq_len=return_seq_len,
                num_inference_steps=num_inference_steps,
                generator=generator,
                device=device,
                known_latents=tmp_known_latents,
                timestamps=timestamps,
                **sampler_kwargs,
            )
        elif sampler_type == "pipeline":
            tmp_samples = pipeline(
                batch_size=num_sample,
                return_seq_len=return_seq_len,
                num_inference_steps=num_inference_steps,
                generator=generator,
                known_latents=tmp_known_latents,
                timestamps=timestamps,
                return_dict=False,
                do_edm_style=True,
                **sampler_kwargs,
            )[0]
        samples[count : count + num_sample] = tmp_samples
        count += num_sample
    return samples
