import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from ladcast.dataloader.utils import precompute_mean_std, xarr_to_tensor
from ladcast.dataloader.weather_dataset import weather_dataset_preprocess_batch
from ladcast.models.DCAE import AutoencoderDC


@torch.no_grad()
def encode_latents_and_save_zarr_direct(
    ds,
    vae,
    zarr_path,
    channel_names,
    normalization_param_dict,
    static_conditioning_tensor,
):
    """
    Encodes latents from the provided xarray dataset directly using xarr_to_tensor
    and saves them to a Zarr store with the "time" coordinate.
    """
    # Initialize the Accelerator
    accelerator = Accelerator()

    # Ensure the VAE is in evaluation mode and moved to the correct device
    vae.eval()
    vae = accelerator.prepare(vae)
    device = accelerator.device
    print(f"device: {device}")

    static_conditioning_tensor = static_conditioning_tensor.to(device)

    # Get all times from the dataset
    all_times = ds.time.values
    total_samples = len(all_times)

    # Preallocate a tensor to store all latents
    latents_tensor = torch.full(
        (total_samples, 84, 15, 30), float("nan"), dtype=torch.float32, device="cpu"
    )

    # Process each time step individually
    progress_bar = tqdm(range(total_samples), desc="Encoding Latents")

    for idx in progress_bar:
        # Extract single timestep
        single_time = all_times[idx]
        ds_single = ds.sel(time=[single_time])

        # Convert to tensor using xarr_to_tensor
        input_tensor = xarr_to_tensor(
            ds_single,
            variable_names=channel_names,
            add_static=False,
            normalization_param_dict=normalization_param_dict,
        ).permute(1, 0, 2, 3)  # shape: (1, C, H, W)

        # Move to device
        input_tensor = input_tensor.to(device)

        # Encode using the VAE
        latent = vae.encode(
            input_tensor,
            static_conditioning_tensor=static_conditioning_tensor.unsqueeze(0),
        ).latent

        # Move latents to CPU and detach from the computation graph
        latent_cpu = latent.squeeze(0).detach().cpu()

        # Assign the latent to the preallocated tensor
        latents_tensor[idx] = latent_cpu

    # Convert the preallocated tensor to a NumPy array
    all_latents = latents_tensor.numpy()

    # Create an xarray Dataset with latents and time as coordinates
    ds_latents = xr.Dataset(
        {"latents": (("time", "C", "H", "W"), all_latents)}, coords={"time": all_times}
    )

    # Save to Zarr
    ds_latents.to_zarr(
        zarr_path,
        mode="w",
    )

    print(f"Latents and time successfully saved to {zarr_path}")

    # Ensure all processes wait until the Zarr file is written
    accelerator.wait_for_everyone()


@torch.no_grad()
def encode_latents_and_save_zarr_hf_dataset(
    zarr_path,
    vae,
    channel_names,
    normalization_param_dict,
    static_conditioning_tensor,
    start_date,
    end_date,
    dataset_py_path="dataloader/weather_dataset.py",
):
    """
    Encodes latents using HuggingFace datasets iterable format and saves to Zarr.

    Args:
        zarr_path (str): Path to save the encoded latents.
        vae (torch.nn.Module): The VAE model.
        normalization_param_dict (dict): Normalization parameters.
        static_conditioning_tensor (torch.Tensor): Static conditioning tensor.
        start_date (str): Start date in format 'YYYY-MM-DDThh'.
        end_date (str): End date in format 'YYYY-MM-DDThh'.
    """
    # Initialize the Accelerator
    accelerator = Accelerator(mixed_precision="no")

    # Ensure the VAE is in evaluation mode and moved to the correct device
    vae.eval()
    vae = accelerator.prepare(vae)
    device = accelerator.device
    print(f"device: {device}")

    # Move static tensor to device
    static_conditioning_tensor = static_conditioning_tensor.to(device)

    # Load the dataset using HuggingFace datasets
    dataset = load_dataset(
        dataset_py_path,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    # Prepare mean and std tensors
    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, variable_names=channel_names
    )
    mean_tensor = mean_tensor[:, None, None].to(device)  # (C, 1, 1)
    std_tensor = std_tensor[:, None, None].to(device)  # (C, 1, 1)

    # Convert to torch format
    dataset = dataset.with_format("torch")

    # Calculate number of timesteps (assuming 1h interval)
    start_dt = datetime.strptime(start_date, "%Y-%m-%dT%H")
    end_dt = datetime.strptime(end_date, "%Y-%m-%dT%H")
    delta = end_dt - start_dt
    total_hours = delta.days * 24 + delta.seconds // 3600 + 1  # +1 to include end date

    print(f"Processing {total_hours} timesteps from {start_date} to {end_date}")

    # Generate all expected timestamps as datetime objects
    all_times = [start_dt + timedelta(hours=i) for i in range(total_hours)]
    # Convert list of datetime objects to a NumPy datetime64 array with nanosecond precision
    all_times_dt64 = np.array(all_times, dtype="datetime64[ns]")

    # Preallocate tensor for latents
    latents_tensor = torch.full(
        (total_hours, 84, 15, 30), float("nan"), dtype=torch.float32, device="cpu"
    )

    # Initialize progress bar
    progress_bar = tqdm(range(total_hours), desc="Encoding Latents")

    # Process each sample
    for idx, sample in enumerate(dataset):
        if idx >= total_hours:
            break

        # Extract data
        input_tensor = sample["data"].to(device)
        # timestamp = sample["timestamp"]

        # Preprocess batch
        input_tensor, nan_mask = weather_dataset_preprocess_batch(
            input_tensor.unsqueeze(0),  # Add batch dimension
            mean_tensor,
            std_tensor,
            crop_south_pole=True,
            sst_channel_idx=82,
            incl_sur_pressure=False,
        )

        # Expand static conditioning tensor
        B = input_tensor.shape[0]
        static_expanded = static_conditioning_tensor.expand(B, -1, -1, -1).clone()
        latent = vae.encode(
            input_tensor, static_conditioning_tensor=static_expanded
        ).latent
        latent_cpu = latent.squeeze(0).detach().cpu()

        # Store latent
        latents_tensor[idx] = latent_cpu

        # Update progress bar
        progress_bar.update(1)

    # Convert to numpy array
    all_latents = latents_tensor.numpy()

    # Create xarray Dataset using datetime64[ns] for time coordinate
    ds_latents = xr.Dataset(
        {"latents": (("time", "C", "H", "W"), all_latents)},
        coords={"time": all_times_dt64},
    )

    # Save to Zarr
    ds_latents.to_zarr(
        zarr_path,
        mode="w",
    )

    print(f"Latents and time successfully saved to {zarr_path}")
    accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Encode data to latent space and save to Zarr")
    
    parser.add_argument("--config", type=str, default="configs/encode_dataloader.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, 
                        help="Path to save the encoded latents (overrides config)")
    parser.add_argument("--ds_path", type=str,
                        help="Path to input dataset (overrides config)")
    parser.add_argument("--start_date", type=str, 
                        help="Start date in format 'YYYY-MM-DDThh' (overrides config)")
    parser.add_argument("--end_date", type=str,
                        help="End date in format 'YYYY-MM-DDThh' (overrides config)")
    parser.add_argument("--repo_name", type=str, default="tonyzyl/ladcast",
                        help="HuggingFace repo name for model")
    parser.add_argument("--model_name", type=str, default="V0.1.X/DCAE",
                        help="Model name/subfolder in repo")
    parser.add_argument("--normalization_json", type=str, default="static/ERA5_normal_1979_2017.json",
                        help="Path to normalization parameters JSON")
    parser.add_argument("--lsm_path", type=str, default="static/240x121_land_sea_mask.pt",
                        help="Path to land-sea mask tensor")
    parser.add_argument("--orography_path", type=str, default="static/240x121_orography.pt",
                        help="Path to orography tensor")
    parser.add_argument("--method", type=str, choices=["direct", "hf"], default="hf",
                        help="Encoding method: 'direct' for xarray or 'hf' for HuggingFace datasets")
    parser.add_argument("--dataset_py_path", type=str, default="dataloader/weather_dataset.py",
                        help="Path to dataset script for HF method")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    full_dataloader_config = config.pop("full_dataloader", OmegaConf.create())
    
    # Get data path from config or args
    ds_path = args.ds_path or full_dataloader_config.get("ds_path")
    start_date = args.start_date or full_dataloader_config.get("start_date")
    end_date = args.end_date or full_dataloader_config.get("end_date")
    
    # Set output path
    zarr_output_path = args.output or full_dataloader_config.get("output_path", "zarr_output_path.zarr")
    
    # Create output directory if needed
    output_dir = Path(zarr_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the xarray dataset directly (if using direct method)
    if args.method == "direct":
        print(f"Loading dataset from {ds_path}")
        ds = xr.open_zarr(ds_path)
        
        # Apply time range selection if specified
        if start_date and end_date:
            ds = ds.sel(time=slice(start_date, end_date))
            
        ds = ds.sel(latitude=slice(-88.5, 90))  # crop south pole
        
        print(f"Dataset loaded with time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Prepare channel names from config
    atmospheric_vars = full_dataloader_config.get("atmospheric_variables_name_list", [])
    surface_vars = full_dataloader_config.get("surface_variables_name_list", [])
    channel_names = atmospheric_vars + surface_vars
    channel_names = list(channel_names)
    
    print(f"Using channels: {channel_names}")
    
    repo_name = args.repo_name
    model_name = args.model_name
    print(f"Loading model from {repo_name}/{model_name}")
    vae = AutoencoderDC.from_pretrained(repo_name, subfolder=model_name)
    
    # Load normalization parameters
    with open(args.normalization_json) as f:
        normalization_param_dict = json.load(f)
    
    # Load and prepare static conditioning tensor
    # TODO: add more checks
    print(f"Loading static conditioning data from {args.lsm_path} and {args.orography_path}")
    lsm_tensor = torch.load(args.lsm_path, weights_only=True)  # (lat, lon)
    lsm_tensor = lsm_tensor[1:, :]  # Crop south pole if needed
    orography_tensor = torch.load(args.orography_path, weights_only=True)  # shape: (4, lat, lon)
    orography_tensor = orography_tensor[:, 1:, :]  # Crop the south pole (first row)
    static_conditioning_tensor = lsm_tensor.unsqueeze(0)  # Shape becomes (1, lat, lon)
    static_conditioning_tensor = torch.cat([static_conditioning_tensor, orography_tensor], dim=0)
    static_mean_tensor = static_conditioning_tensor.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
    static_std_tensor = static_conditioning_tensor.std(dim=(1, 2), keepdim=True)
    static_conditioning_tensor = (static_conditioning_tensor - static_mean_tensor) / static_std_tensor
    
    print(f"Encoding data using {args.method} method and saving to {zarr_output_path}")
    
    # Encode latents and save to Zarr using the specified approach
    if args.method == "direct":
        encode_latents_and_save_zarr_direct(
            ds=ds,
            vae=vae,
            zarr_path=zarr_output_path,
            channel_names=channel_names,
            normalization_param_dict=normalization_param_dict,
            static_conditioning_tensor=static_conditioning_tensor
        )
    else:  # args.method == "hf"
        if not start_date or not end_date:
            raise ValueError("start_date and end_date must be provided for HF dataset method")
            
        encode_latents_and_save_zarr_hf_dataset(
            zarr_path=zarr_output_path,
            vae=vae,
            channel_names=channel_names,
            normalization_param_dict=normalization_param_dict,
            static_conditioning_tensor=static_conditioning_tensor,
            start_date=start_date,
            end_date=end_date,
            dataset_py_path=args.dataset_py_path,
        )


if __name__ == "__main__":
    """
    Example usage:
    python -m ladcast.preprocess.encode_data --method direct --output ./outputs/latents.zarr
    python -m ladcast.preprocess.encode_data --method hf --start_date 2020-01-01T00 --end_date 2020-01-31T23
    """
    main()