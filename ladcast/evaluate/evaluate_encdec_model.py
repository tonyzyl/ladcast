import argparse
import json

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ladcast.dataloader.utils import precompute_mean_std
from ladcast.dataloader.weather_dataset import weather_dataset_preprocess_batch
from ladcast.evaluate.utils import get_normalized_lat_weights_based_on_cos
from ladcast.metric.loss import LpLoss
from ladcast.metric.utils import process_tensor_for_loss
from ladcast.models.DCAE import AutoencoderDC

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate encdec model based on IterableDataset."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--normalization_json",
        type=str,
        default="ERA5_normal.json",
        help="Path to the normalization JSON file",
    )
    parser.add_argument(
        "--lsm_path", type=str, default=None, help="Path to the land-sea mask tensor."
    )
    parser.add_argument(
        "--orography_path", type=str, default=None, help="Path to the orography tensor."
    )
    parser.add_argument(
        "--iter_buffer_size",
        type=int,
        default=100,
        help="The buffer size for the initialized random buffer.",
    )
    parser.add_argument(
        "--encdec_cls",
        type=str,
        default="dcae",
        choices=["dcae"],
        help="The class of the encdec model.",
    )
    parser.add_argument(
        "--encdec_model",
        type=str,
        default="V0.1.X/DCAE",
        help="The name of the encdec model subfolder in the repo.",
    )
    parser.add_argument(
        "--csv_path", type=str, default=None, help="Path to the CSV file for logging."
    )
    return parser.parse_args()


def main(args):
    # ─── load configs & set up ─────────────────────────────────────────────
    config = OmegaConf.load(args.config)
    val_dataloader_config = config.pop("val_dataloader", OmegaConf.create())
    general_config = config.pop("general", OmegaConf.create())

    loss_fn = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")

    # normalization stats
    if args.normalization_json:
        with open(args.normalization_json, "r") as f:
            normalization_param_dict = json.load(f)
    else:
        normalization_param_dict = None

    # Accelerator
    accelerator = Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True),
    )

    # ─── load pretrained model ─────────────────────────────────────────────
    encdec_model_cls = AutoencoderDC
    repo_name = "tonyzyl/ladcast"
    encdec_model = encdec_model_cls.from_pretrained(
        pretrained_model_name_or_path=repo_name, subfolder=args.encdec_model
    )
    # encdec = encdec_model_cls.from_pretrained(args.encdec_model)
    encdec_model = accelerator.prepare(encdec_model)
    encdec_model.eval()
    encdec_model.requires_grad_(False)

    # ─── prepare validation datasets & dataloaders ─────────────────────────
    splits = ["2018", "2019", "2020", "2021", "2022"]
    val_loaders = {}
    for year in splits:
        ds = load_dataset(
            "dataloader/weather_dataset.py",
            split=year,
            streaming=True,
            trust_remote_code=True,
        ).with_format("torch")
        dl = DataLoader(ds, **val_dataloader_config)
        val_loaders[year] = accelerator.prepare(dl)

    # ─── compute mean/std & static conditioning ───────────────────────────
    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, variable_names=general_config.channel_names
    )
    mean_tensor = mean_tensor[:, None, None].to(accelerator.device)
    std_tensor = std_tensor[:, None, None].to(accelerator.device)

    static_conditioning_tensor = None
    num_static_tensors = 0
    if args.lsm_path:
        lsm = torch.load(args.lsm_path, weights_only=True).to(accelerator.device)[1:, :]
        static_conditioning_tensor = lsm.unsqueeze(0)
    if args.orography_path:
        oro = torch.load(args.orography_path, weights_only=True).to(accelerator.device)[
            :, 1:, :
        ]
        static_conditioning_tensor = (
            oro
            if static_conditioning_tensor is None
            else torch.cat([static_conditioning_tensor, oro], dim=0)
        )
    if static_conditioning_tensor is not None:
        static_mean = static_conditioning_tensor.mean((1, 2), keepdim=True)
        static_std = static_conditioning_tensor.std((1, 2), keepdim=True)
        static_conditioning_tensor = (
            (static_conditioning_tensor - static_mean) / static_std
        ).to(accelerator.device)
        num_static_tensors = static_conditioning_tensor.shape[0]

    # ─── latitude weights for lat‐weighted MSE ─────────────────────────────
    latitude = np.linspace(-88.5, 90, 120)
    lat_weight = get_normalized_lat_weights_based_on_cos(latitude)
    lat_weight = torch.from_numpy(lat_weight).to(accelerator.device).unsqueeze(1)

    # ─── constants ─────────────────────────────────────────────────────────
    PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    NUM_ATM_VARS = 6
    NUM_SUR_VARS = 6

    # ─── main evaluation loop ──────────────────────────────────────────────
    all_yearly_logs = []
    with torch.no_grad():
        for year, loader in val_loaders.items():
            # reset accumulators
            val_channel_lw_mse = torch.zeros(
                NUM_ATM_VARS * len(PRESSURE_LEVELS) + NUM_SUR_VARS + num_static_tensors,
                device=accelerator.device,
            )
            val_loss_fn_loss = 0.0
            val_aug_loss = 0.0
            total_val_size = 0

            for batch in tqdm(loader, desc=f"Validating {year}"):
                # timestamp debug
                ts = batch["timestamp"]
                print(f"[rank{accelerator.process_index}] Year {year}, timestamp {ts}")

                # preprocess (mask nans, crop south pole, normalize)
                batch, nan_mask = weather_dataset_preprocess_batch(
                    batch["data"],
                    mean_tensor,
                    std_tensor,
                    crop_south_pole=True,
                    sst_channel_idx=82,
                    incl_sur_pressure=False,
                )

                B, C, H, W = batch.shape
                static_expanded = (
                    static_conditioning_tensor.expand(B, -1, -1, -1)
                    if static_conditioning_tensor is not None
                    else None
                )

                # forward
                pred = encdec_model(
                    batch,
                    return_static=True,
                    static_conditioning_tensor=static_expanded,
                ).sample

                # loss‐masking
                pred, input_tensor = process_tensor_for_loss(
                    pred, batch, nan_mask, sst_chanel_idx=82
                )
                if args.encdec_cls == "dcae":
                    input_tensor = torch.cat([input_tensor, static_expanded], dim=1)

                # augmentation metric
                aug_loss = 0.0

                # custom LpLoss
                tmp_fn = loss_fn(
                    pred, input_tensor, weight=lat_weight.view(1, 1, -1, 1)
                )
                val_loss_fn_loss += accelerator.gather(tmp_fn).mean().item() * B

                # unnormalize & compute per‐channel MSE
                proc_mean = (
                    torch.cat([mean_tensor, static_mean], dim=0)
                    if num_static_tensors > 0
                    else mean_tensor
                )
                proc_std = (
                    torch.cat([std_tensor, static_std], dim=0)
                    if num_static_tensors > 0
                    else std_tensor
                )
                mse_map = torch.nn.functional.mse_loss(
                    pred * proc_std + proc_mean,
                    input_tensor * proc_std + proc_mean,
                    reduction="none",
                )
                mse_map = accelerator.gather(mse_map)

                # lat‐weighted
                lw = (mse_map * lat_weight.view(1, 1, -1, 1)).mean(dim=[0, 2, 3])
                val_channel_lw_mse += lw * B

                val_aug_loss += aug_loss * B
                total_val_size += B

            # finalize this year
            val_loss_fn_loss /= total_val_size
            val_channel_lw_mse /= total_val_size
            val_channel_lw_rmse = torch.sqrt(val_channel_lw_mse)

            if accelerator.is_main_process:
                logs = {"year": year, "val_loss_fn_loss": val_loss_fn_loss}
                # atm levels
                for vi in range(NUM_ATM_VARS):
                    for pi, p in enumerate(PRESSURE_LEVELS):
                        key = (
                            f"val_lw_rmse_{general_config.channel_names[vi]}_level_{p}"
                        )
                        logs[key] = val_channel_lw_rmse[
                            vi * len(PRESSURE_LEVELS) + pi
                        ].item()
                # surface vars
                base = NUM_ATM_VARS * len(PRESSURE_LEVELS)
                for si in range(NUM_SUR_VARS):
                    key = (
                        f"val_lw_rmse_{general_config.channel_names[NUM_ATM_VARS + si]}"
                    )
                    logs[key] = val_channel_lw_rmse[base + si].item()
                # static (if any)
                for sti in range(num_static_tensors):
                    key = f"val_lw_rmse_{general_config.static_names[sti]}"
                    logs[key] = val_channel_lw_rmse[base + NUM_SUR_VARS + sti].item()
                # aug loss

                all_yearly_logs.append(logs)

    # ─── write out CSV ─────────────────────────────────────────────────────
    if accelerator.is_main_process:
        result_df = pd.DataFrame(all_yearly_logs)
        result_df.to_csv(args.csv_path, index=False)
        logger.info(f"Saved per‐year validation metrics to {args.csv_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
