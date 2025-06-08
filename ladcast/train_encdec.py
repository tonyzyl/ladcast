import argparse
import json
import logging
import math
import os
import pprint
import shutil
from datetime import timedelta
from io import StringIO
from pathlib import Path

import accelerate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DataLoaderConfiguration,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import upload_folder
from omegaconf import OmegaConf
from packaging import version
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ladcast.dataloader.utils import periodic_rearrange_batch, precompute_mean_std
from ladcast.dataloader.weather_dataset import weather_dataset_preprocess_batch
from ladcast.evaluate.utils import get_normalized_lat_weights_based_on_cos
from ladcast.metric.utils import process_tensor_for_loss, remove_channel
from ladcast.models.DCAE import AutoencoderDC
from ladcast.utils import flatten_and_filter_config, instantiate_from_config

if is_wandb_available():
    pass

logger = get_logger(__name__, log_level="INFO")


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
        "--resume_from_path",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous saved path. Model loading is done by from_pretrained."
        ),
    )
    parser.add_argument(
        "--resume_from_hub",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous saved path on the Hub. Model loading is done by from_pretrained."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--norm_json_path",
        type=str,
        default=None,
        help=("Path to the JSON file containing the normalization statistics."),
    )
    parser.add_argument(
        "--lsm_path", type=str, default=None, help="Path to the land-sea mask tensor."
    )
    parser.add_argument(
        "--orography_path", type=str, default=None, help="Path to the orography tensor."
    )
    parser.add_argument(
        "--lat_weighted_loss",
        action="store_true",
        help="Whether to use latitudinally weighted loss.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--ft_decoder", action="store_true", help="Whether to fine-tune the decoder."
    )
    parser.add_argument(
        "--encdec_cls", type=str, default="cnn", help="The encdec model class to use."
    )  # LaDCast
    parser.add_argument(
        "--aug_metric", type=str, default=None, help="The augmentation metric to use."
    )
    parser.add_argument(
        "--iter_buffer_size",
        type=int,
        default=10000,
        help="The buffer size for the initialized random buffer.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    return parser.parse_args()


def main(args):
    # workflow based on: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

    config = OmegaConf.load(args.config)
    tracker_config = flatten_and_filter_config(
        OmegaConf.to_container(config, resolve=True)
    )

    encdec_config = OmegaConf.to_container(
        config.pop("encdec", OmegaConf.create()), resolve=True
    )
    accelerator_config = config.pop("accelerator", OmegaConf.create())
    loss_fn_config = config.pop("loss_fn", OmegaConf.create())
    loss_scale_config = config.pop("loss_scale", OmegaConf.create())
    optimizer_config = config.pop("optimizer", OmegaConf.create())
    lr_scheduler_config = config.pop("lr_scheduler", OmegaConf.create())
    train_dataloader_config = config.pop("train_dataloader", OmegaConf.create())
    val_dataloader_config = config.pop("val_dataloader", OmegaConf.create())
    ema_config = config.pop("ema", OmegaConf.create())
    general_config = config.pop("general", OmegaConf.create())

    if args.norm_json_path:
        with open(args.norm_json_path, "r") as f:
            normalization_param_dict = json.load(f)
    else:
        normalization_param_dict = None

    logging_dir = Path(general_config.output_dir, general_config.logging_dir)
    # https://github.com/huggingface/transformers/issues/24361
    accelerator_dataloader_config = DataLoaderConfiguration(
        dispatch_batches=True
    )  # False for using IterableDataset instead of DataLoaderDispatcher.
    accelerator_project_config = ProjectConfiguration(
        project_dir=general_config.output_dir, logging_dir=logging_dir
    )
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    kwargs_list = [kwargs]
    kwargs_list = []
    accelerator = Accelerator(
        project_config=accelerator_project_config,
        dataloader_config=accelerator_dataloader_config,
        **OmegaConf.to_container(accelerator_config, resolve=True),
        kwargs_handlers=kwargs_list,
    )

    set_seed(general_config.seed, device_specific=True)

    # encdec_model_cls = AutoencoderKL
    if args.encdec_cls == "dcae":
        encdec_model_cls = AutoencoderDC
    else:
        raise ValueError(f"Unknown encdec model class: {args.encdec_cls}")
    if args.resume_from_path:
        print(f"Loading model from {args.resume_from_path}")
        encdec = encdec_model_cls.from_pretrained(args.resume_from_path)
    elif args.resume_from_hub:
        print(f"Loading model from {args.resume_from_hub}")
        repo_name = args.hub_model_id
        encdec = encdec_model_cls.from_pretrained(
            repo_name, subfolder=args.resume_from_hub
        )
    else:
        print("Training from scratch")
        encdec = encdec_model_cls.from_config(config=encdec_config)

    if args.ft_decoder:
        print("Freezing the encoder, only Fine-tuning the decoder")
        encdec.encoder.requires_grad_(False)
        encdec.decoder.requires_grad_(True)

    # Create EMA for the model. Some examples:
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
    if ema_config.use_ema:
        ema_model = EMAModel(
            encdec.parameters(),
            decay=ema_config.ema_max_decay,
            use_ema_warmup=True,
            update_after_step=ema_config.ema_update_after_step,
            inv_gamma=ema_config.ema_inv_gamma,
            power=ema_config.ema_power,
            model_cls=encdec_model_cls,
            model_config=encdec.config,
            foreach=ema_config.foreach,
        )

    sub_folder_name = "encdec"
    sub_folder_ema_name = "encdec_ema"

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if ema_config.use_ema:
                    ema_model.save_pretrained(
                        os.path.join(output_dir, sub_folder_ema_name)
                    )

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, sub_folder_name))

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
                    os.path.join(input_dir, sub_folder_ema_name),
                    encdec_model_cls,  # , foreach=ema_config.foreach # not yet released in v0.29.2
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
                load_model = encdec_model_cls.from_pretrained(
                    input_dir, subfolder=sub_folder_name
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    loss_fn = instantiate_from_config(loss_fn_config)
    # aug_metric_cls = None
    # aug_metric_weight = 0.0

    """
    generator = torch.Generator(device='cpu').manual_seed(general_config.seed)
    with accelerator.main_process_first():
        # https://github.com/huggingface/accelerate/issues/503
        # https://discuss.huggingface.co/t/shared-memory-in-accelerate/28619
        train_dataloader = prepare_dataloader_69var(**train_dataloader_config, normalization_param_dict=normalization_param_dict)
        val_dataloader = prepare_dataloader_69var(**val_dataloader_config, normalization_param_dict=normalization_param_dict)
        #train_dataloader = create_random_tensor_dataloader(batch_size=1, num_samples=1000, channels=69, height=720, width=1440)
        #val_dataloader = create_random_tensor_dataloader(batch_size=1, num_samples=100, channels=69, height=720, width=1440)
    """
    train_dataset = load_dataset(
        "dataloader/weather_dataset.py",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    train_dataset = train_dataset.shuffle(
        seed=general_config.seed, buffer_size=args.iter_buffer_size
    )
    train_dataset = train_dataset.with_format("torch")
    val_dataset = load_dataset(
        "dataloader/weather_dataset.py",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    val_dataset = val_dataset.with_format("torch")

    train_dataloader = DataLoader(train_dataset, **train_dataloader_config)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_config)
    mean_tensor, std_tensor = precompute_mean_std(
        normalization_param_dict, variable_names=general_config.channel_names
    )
    mean_tensor = mean_tensor[:, None, None]  # -> (C, 1, 1)
    std_tensor = std_tensor[:, None, None]  # -> (C, 1, 1)
    mean_tensor = mean_tensor.to(accelerator.device)
    std_tensor = std_tensor.to(accelerator.device)

    if args.lsm_path:
        lsm_tensor = torch.load(args.lsm_path, weights_only=True).to(
            accelerator.device
        )  # (lat, lon)
        lsm_tensor = lsm_tensor[1:, :]  # crop south pole (first row)
    if args.orography_path:
        # ['standard_deviation_of_orography', 'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography', 'slope_of_sub_gridscale_orography']
        orography_tensor = torch.load(args.orography_path, weights_only=True).to(
            accelerator.device
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
        ).to(accelerator.device)  # (C, 1, 1)
        static_std_tensor = static_conditioning_tensor.std(dim=(1, 2), keepdim=True).to(
            accelerator.device
        )
        static_conditioning_tensor = (
            (static_conditioning_tensor - static_mean_tensor) / static_std_tensor
        ).to(accelerator.device)
        if args.encdec_cls == "dcae":
            # only dcae do static conditioning tensor reconstruction
            num_static_tensors = static_conditioning_tensor.shape[0]
        else:
            raise NotImplementedError
    else:
        num_static_tensors = 0

    # Make one log on every process with the configuration for debugging.
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

    if args.gradient_checkpointing:
        encdec.enable_gradient_checkpointing()

    optimizer = AdamW(
        encdec.parameters(), **OmegaConf.to_container(optimizer_config, resolve=True)
    )

    if "subbatch_steps" in general_config:
        num_subbatch_steps = int(
            general_config.subbatch_steps
        )  # num_steps to augment the batch
    else:
        num_subbatch_steps = 1
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = (
        lr_scheduler_config.num_warmup_steps
        * accelerator.num_processes
        * num_subbatch_steps
    )
    # Scheduler and math around the number of training steps.

    # Before accelerate.prepare len(dataloader) -> len(dataset) / batch_size
    if (
        "num_training_steps" not in general_config
        or general_config.num_training_steps is None
    ):
        if isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
            # for details, check IterableDatasetShard (dispatch_batches=False) implementation in accelerate
            # In summary, it has the set_epoch() for updating the rng,
            # split_batch=False(default) -> run num_processes iter for getting the batch at a single training step
            assert general_config.epoch_length is not None, (
                "You need to set 'epoch_length' in the config for IterableDataset"
            )
            len_train_dataloader = math.ceil(
                general_config.epoch_length / train_dataloader_config.batch_size
            )
            len_train_dataloader_after_sharding = math.ceil(
                len_train_dataloader / accelerator.num_processes
            )
        else:
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

    encdec, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            encdec, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    )

    if ema_config.use_ema:
        if ema_config.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # After accelerator.prepare, len(dataloader) -> len(dataset) / batch_size / num_processes
    if isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
        len_train_dataloader = math.ceil(
            math.ceil(general_config.epoch_length / train_dataloader_config.batch_size)
            / accelerator.num_processes
        )
    else:
        len_train_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = (
        math.ceil(len_train_dataloader / accelerator.gradient_accumulation_steps)
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
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len_train_dataloader}) does not match "
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
        buffer = StringIO()
        pprint.pp(tracker_config, stream=buffer)
        logger.info(f"Tracker configuration:\n{buffer.getvalue()}")
        # run_name = general_config.output_dir.split("/")[-1]
        accelerator.init_trackers(
            general_config.tracker_project_name, config=tracker_config
        )

    total_batch_size = (
        train_dataloader_config.batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        # https://github.com/huggingface/diffusers/issues/6503
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    if isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
        logger.info(
            f"  Num batches each epoch = {math.ceil(general_config.epoch_length / train_dataloader_config.batch_size)}"
        )
    else:
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
    logger.info(
        f"  Num additional sub batch with data augmentation = {num_subbatch_steps - 1}"
    )
    logger.info(f"  Total optimization steps = {general_config.num_training_steps}")
    logger.info(f"  Total training epochs = {general_config.num_train_epochs}")
    global_step = 0
    first_epoch = 0

    PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    NUM_ATM_VARS = 6
    NUM_SUR_VARS = 6

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
            accelerator.load_state(os.path.join(general_config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    best_model_path = os.path.join(general_config.output_dir, "best_val")
    os.makedirs(best_model_path, exist_ok=True)

    progress_bar = tqdm(
        range(0, general_config.num_training_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    latitude = np.linspace(-88.5, 90, 120)  # crop south pole
    lat_weight = get_normalized_lat_weights_based_on_cos(latitude)  # (lat,)
    lat_weight = (
        torch.from_numpy(lat_weight).to(accelerator.device).unsqueeze(1)
    )  # (lat, 1)

    best_val_loss = float("inf")
    # Now you train the model
    for epoch in range(first_epoch, general_config.num_train_epochs):
        encdec.train()
        train_loss = 0.0
        train_loss_fn_loss = 0.0
        # train_kl_loss = 0.0
        train_aug_loss = 0.0
        logger.info(f"Starting Epoch {epoch + 1}")
        if isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
            train_dataloader.set_epoch(epoch)
        for _, batch in enumerate(train_dataloader):
            # timestamp = batch["timestamp"]
            # timestamp = timestamp_tensor_to_time_elapsed(timestamp)
            # print('Process: ', accelerator.process_index, batch["timestamp"])

            if isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
                # mask out nan in sst, crop the south pole
                # nan_mask: [B, H, W]
                if args.encdec_cls == "dcae":
                    batch, nan_mask = weather_dataset_preprocess_batch(
                        batch["data"],
                        mean_tensor,
                        std_tensor,
                        crop_south_pole=True,
                        sst_channel_idx=82,
                        incl_sur_pressure=False,
                    )
                else:
                    raise NotImplementedError
            B, C, H, W = batch.shape

            static_expanded = None
            if static_conditioning_tensor is not None:
                static_expanded = static_conditioning_tensor.expand(
                    B, -1, -1, -1
                ).clone()

            if accelerator.num_processes > 1:
                lat_weight_expanded = lat_weight.expand(
                    B, encdec.module.config.out_channels, -1, -1
                ).clone()
            else:
                lat_weight_expanded = lat_weight.expand(
                    B, encdec.config.out_channels, -1, -1
                ).clone()  # (B, C, lat, 1)

            for subbatch_step in range(num_subbatch_steps):
                coords = None
                if subbatch_step > 0:
                    new_x = torch.randint(0, W, (B,))
                    new_y = torch.randint(0, H, (B,))
                    coords = torch.stack([new_x, new_y], dim=1)
                    batch = periodic_rearrange_batch(
                        batch, coords=coords
                    )  # data augmentation
                    nan_mask = periodic_rearrange_batch(
                        nan_mask.unsqueeze(1), coords=coords
                    ).squeeze(1)  # [B, H, W]
                    lat_weight_expanded = periodic_rearrange_batch(
                        lat_weight_expanded, coords=coords
                    )  # (B, C, lat, 1)
                    # print("Process idx: ", accelerator.process_index, " Coords for augmentation: ", coords[0])
                    if static_conditioning_tensor is not None:
                        static_expanded = periodic_rearrange_batch(
                            static_expanded, coords=coords
                        )

                input_tensor = batch.clone()
                with accelerator.accumulate(encdec):
                    # For now, only XAttnDiffusersAutoencoderKL support passing coords
                    if args.encdec_cls == "dcae":
                        pred = encdec(
                            input_tensor,
                            return_static=True,
                            static_conditioning_tensor=static_expanded,
                        ).sample
                    else:
                        raise NotImplementedError

                    if args.encdec_cls == "dcae":
                        # mask values corresponding to nan in SST
                        pred, input_tensor = process_tensor_for_loss(
                            pred, input_tensor, nan_mask, sst_chanel_idx=82
                        )
                        # dcae model has static reconstruction loss
                        input_tensor = torch.cat((input_tensor, static_expanded), dim=1)
                    else:
                        raise NotImplementedError

                    if args.lat_weighted_loss:
                        loss_fn_loss = loss_fn(
                            pred, input_tensor, weight=lat_weight_expanded
                        )
                    else:
                        loss_fn_loss = loss_fn(pred, input_tensor)

                    loss_fn_each_var = loss_fn.get_loss_per_var(
                        pred,
                        input_tensor,
                        num_atm_vars=NUM_ATM_VARS,
                        num_levels=len(PRESSURE_LEVELS),
                    )
                    lw_loss_fn_each_var = loss_fn.get_loss_per_var(
                        pred,
                        input_tensor,
                        num_atm_vars=NUM_ATM_VARS,
                        num_levels=len(PRESSURE_LEVELS),
                        weight=lat_weight_expanded,
                    )
                    # if args.encdec_cls == "dcae":
                    # kl_loss = 0
                    # else:
                    # kl_loss = posterior.kl().mean()
                    aug_loss = 0

                    loss = loss_scale_config.loss_fn_scale * loss_fn_loss + aug_loss

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(
                        loss.repeat(train_dataloader_config.batch_size)
                    ).mean()
                    avg_loss_fn_loss = accelerator.gather(
                        loss_fn_loss.repeat(train_dataloader_config.batch_size)
                    ).mean()
                    # if not args.encdec_cls == "dcae":
                    # avg_kl_loss = accelerator.gather(kl_loss.repeat(train_dataloader_config.batch_size)).mean()
                    # train_kl_loss += avg_kl_loss.item() / accelerator.gradient_accumulation_steps
                    # else:
                    # train_kl_loss = 0
                    train_loss += (
                        avg_loss.item() / accelerator.gradient_accumulation_steps
                    )
                    train_loss_fn_loss += (
                        avg_loss_fn_loss.item()
                        / accelerator.gradient_accumulation_steps
                    )
                    if args.aug_metric is not None:
                        avg_aug_loss = accelerator.gather(
                            aug_loss.repeat(train_dataloader_config.batch_size)
                        ).mean()
                        train_aug_loss += (
                            avg_aug_loss.item()
                            / accelerator.gradient_accumulation_steps
                        )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(encdec.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if ema_config.use_ema:
                        if ema_config.offload_ema:
                            ema_model.to(device="cuda", non_blocking=True)
                        ema_model.step(encdec.parameters())
                        if ema_config.offload_ema:
                            ema_model.to(device="cpu", non_blocking=True)
                    progress_bar.update(1)
                    logs = {
                        "train loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "train_loss_fn": train_loss_fn_loss,
                    }
                    for var_idx in range(
                        loss_fn_each_var.shape[0] - num_static_tensors
                    ):
                        # surface pressure placed at last
                        logs[
                            f"train_loss_fn_{general_config.channel_names[var_idx]}"
                        ] = loss_fn_each_var[var_idx]
                        logs[
                            f"train_lw_loss_fn_{general_config.channel_names[var_idx]}"
                        ] = lw_loss_fn_each_var[var_idx]
                    if args.encdec_cls == "dcae":
                        for var_idx in range(num_static_tensors):
                            logs[
                                f"train_loss_fn_{general_config.static_names[var_idx]}"
                            ] = loss_fn_each_var[
                                var_idx
                                + (loss_fn_each_var.shape[0] - num_static_tensors)
                            ]
                    if args.aug_metric is not None:
                        logs["train_aug_loss"] = train_aug_loss
                    if ema_config.use_ema:
                        logs["ema_decay"] = ema_model.cur_decay_value
                    global_step += 1
                    accelerator.log(logs, step=global_step)
                    train_loss = 0.0
                    train_loss_fn_loss = 0.0
                    # train_kl_loss = 0.0
                    train_aug_loss = 0.0

                    if global_step % general_config.checkpointing_steps == 0:
                        if accelerator.is_main_process:
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
                                        len(checkpoints)
                                        - args.checkpoints_total_limit
                                        + 1
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
                                            general_config.output_dir,
                                            removing_checkpoint,
                                        )
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            general_config.output_dir, f"checkpoint-{global_step}"
                        )
                        logger.info(f"Logging state to {save_path}")
                        accelerator.save_state(
                            save_path
                        )  # From accelerate official guide, distributed saving needs to be done outside to save all states

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "loss_fn_loss": loss_fn_loss.detach().item(),
                }

                if ema_config.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                if args.aug_metric is not None:
                    logs["aug_loss"] = aug_loss.detach().item()
                progress_bar.set_postfix(**logs)

                if global_step >= general_config.num_training_steps:
                    break

        if (
            (epoch + 1) % general_config.val_every_epochs == 0
            or epoch == general_config.num_train_epochs - 1
        ):
            logger.info("Starting validation")
            if ema_config.use_ema:
                ema_model.store(encdec.parameters())
                ema_model.copy_to(encdec.parameters())
            encdec.eval()
            val_loss = 0.0
            val_loss_fn_loss = 0.0
            # val_kl_loss = 0.0
            val_aug_loss = 0.0
            total_val_size = 0
            if accelerator.num_processes > 1:
                val_channel_mse = torch.zeros(
                    encdec.module.config.out_channels,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                val_channel_lw_mse = torch.zeros(
                    encdec.module.config.out_channels,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
            else:
                val_channel_mse = torch.zeros(
                    encdec.config.out_channels,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                val_channel_lw_mse = torch.zeros(
                    encdec.config.out_channels,
                    device=accelerator.device,
                    dtype=torch.float32,
                )

            with torch.no_grad():  # Disable gradient calculations
                for batch in val_dataloader:
                    # timestamp = batch["timestamp"]
                    # timestamp = timestamp_tensor_to_time_elapsed(timestamp)

                    if isinstance(
                        val_dataloader.dataset, torch.utils.data.IterableDataset
                    ):
                        # mask out nan in sst, crop the south pole
                        if args.encdec_cls == "dcae":
                            batch, nan_mask = weather_dataset_preprocess_batch(
                                batch["data"],
                                mean_tensor,
                                std_tensor,
                                crop_south_pole=True,
                                sst_channel_idx=82,
                                incl_sur_pressure=False,
                            )
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

                    B, C, H, W = batch.shape

                    static_expanded = None
                    if static_conditioning_tensor is not None:
                        static_expanded = static_conditioning_tensor.expand(
                            B, -1, -1, -1
                        ).clone()

                    input_tensor = batch  # no need to clone, we have no substep at val
                    if args.encdec_cls == "dcae":
                        pred = encdec(
                            input_tensor,
                            return_static=True,
                            static_conditioning_tensor=static_expanded,
                        ).sample

                        pred, input_tensor = process_tensor_for_loss(
                            pred, input_tensor, nan_mask, sst_chanel_idx=82
                        )
                    else:
                        raise NotImplementedError

                    if args.encdec_cls == "dcae":
                        # dcae model has static reconstruction loss
                        input_tensor = torch.cat((input_tensor, static_expanded), dim=1)

                    if args.lat_weighted_loss:
                        loss_fn_loss = loss_fn(
                            pred, input_tensor, weight=lat_weight.view(1, 1, -1, 1)
                        )  # Compute reconstruction loss
                    else:
                        loss_fn_loss = loss_fn(pred, input_tensor)
                    # if args.encdec_cls == "dcae":
                    # kl_loss = 0
                    # else:
                    # kl_loss = posterior.kl().mean()
                    aug_loss = 0

                    loss = loss_scale_config.loss_fn_scale * loss_fn_loss + aug_loss

                    if args.encdec_cls == "transformer":
                        processed_mean_tensor = remove_channel(
                            mean_tensor.unsqueeze(0),
                            encdec_config["sur_pressure_var_idx"],
                        ).squeeze(0)
                        processed_std_tensor = remove_channel(
                            std_tensor.unsqueeze(0),
                            encdec_config["sur_pressure_var_idx"],
                        ).squeeze(0)
                    else:
                        processed_mean_tensor = mean_tensor
                        processed_std_tensor = std_tensor

                    if args.encdec_cls == "dcae":
                        processed_mean_tensor = torch.cat(
                            (processed_mean_tensor, static_mean_tensor), dim=0
                        )
                        processed_std_tensor = torch.cat(
                            (processed_std_tensor, static_std_tensor), dim=0
                        )

                    # unnormalize before computing mse
                    tmp_cur_batch_mse = torch.nn.functional.mse_loss(
                        pred * processed_std_tensor + processed_mean_tensor,
                        input_tensor * processed_std_tensor + processed_mean_tensor,
                        reduction="none",
                    )
                    cur_batch_mse = accelerator.gather(tmp_cur_batch_mse)

                    # compute lat-weighted mse
                    cur_batch_lw_mse = cur_batch_mse * lat_weight.view(1, 1, -1, 1)
                    cur_batch_lw_mse = cur_batch_lw_mse.mean(
                        dim=[0, 2, 3]
                    )  # mean over batch, height, width
                    val_channel_lw_mse += cur_batch_lw_mse * input_tensor.shape[0]

                    cur_batch_mse = cur_batch_mse.mean(
                        dim=[0, 2, 3]
                    )  # mean over batch, height, width
                    val_channel_mse += cur_batch_mse * input_tensor.shape[0]

                    # Accumulate the validation loss
                    val_loss += loss.item() * input_tensor.shape[0]
                    val_loss_fn_loss += loss_fn_loss.item() * input_tensor.shape[0]
                    # if args.encdec_cls != "dcae":
                    # val_kl_loss += kl_loss.item() * batch.shape[0]
                    # else:
                    # val_kl_loss = 0
                    if args.aug_metric is not None:
                        val_aug_loss += aug_loss.item() * input_tensor.shape[0]
                    total_val_size += input_tensor.shape[0]

            # Average validation loss over all batches
            val_loss /= total_val_size
            val_loss_fn_loss /= total_val_size
            # val_kl_loss /= total_val_size
            val_aug_loss /= total_val_size
            val_channel_mse /= total_val_size
            val_channel_rmse = torch.sqrt(val_channel_mse)
            val_channel_lw_mse /= total_val_size
            val_channel_lw_rmse = torch.sqrt(val_channel_lw_mse)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # temp fix for incompatibility with torchdynamo & unwrap_model
                # accelerator.unwrap_model(encdec).save_pretrained(best_model_path)

                if accelerator.is_main_process:
                    val_checkpoints = os.listdir(best_model_path)
                    val_checkpoints = [
                        d for d in val_checkpoints if d.startswith("epoch")
                    ]
                    val_checkpoints = sorted(
                        val_checkpoints, key=lambda x: int(x.split("-")[1])
                    )

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(val_checkpoints) >= 3:
                        num_to_remove = len(val_checkpoints) - 3 + 1
                        removing_val_checkpoints = val_checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(val_checkpoints)} val checkpoints already exist, removing {len(removing_val_checkpoints)} val checkpoints"
                        )
                        logger.info(
                            f"removing val checkpoints: {', '.join(removing_val_checkpoints)}"
                        )

                        for removing_val_checkpoint in removing_val_checkpoints:
                            removing_val_checkpoint = os.path.join(
                                best_model_path, removing_val_checkpoint
                            )
                            shutil.rmtree(removing_val_checkpoint)

                save_path = os.path.join(best_model_path, f"epoch-{epoch + 1}")
                logger.info(
                    f"Best val found on epoch {epoch + 1}. Saving model at {save_path}"
                )
                accelerator.save_state(
                    save_path
                )  # From accelerate official guide, distributed saving needs to be done outside to save all states

            if accelerator.is_main_process:
                # skip logging image for now
                # log_validation('val', general_config, int(epoch+1), unwrap_model(encdec), batch, mean_tensor, std_tensor, accelerator)
                logger.info(
                    f"Validation loss after epoch {epoch + 1}: {val_loss}, loss_fn_loss: {val_loss_fn_loss}, aug_loss: {val_aug_loss}"
                )
                logs = {
                    "val loss": val_loss,
                    " val_loss_fn": val_loss_fn_loss,
                    "step": global_step,
                }
                for var_idx in range(NUM_ATM_VARS):
                    # surface pressure placed at last
                    for level_idx in range(len(PRESSURE_LEVELS)):
                        logs[
                            f"val_rmse_{general_config.channel_names[var_idx]}_level_{PRESSURE_LEVELS[level_idx]}"
                        ] = val_channel_rmse[var_idx * len(PRESSURE_LEVELS) + level_idx]
                        logs[
                            f"val_lw_rmse_{general_config.channel_names[var_idx]}_level_{PRESSURE_LEVELS[level_idx]}"
                        ] = val_channel_lw_rmse[
                            var_idx * len(PRESSURE_LEVELS) + level_idx
                        ]
                for var_idx in range(NUM_SUR_VARS):
                    logs[
                        f"val_rmse_{general_config.channel_names[var_idx + NUM_ATM_VARS]}"
                    ] = val_channel_rmse[
                        (NUM_ATM_VARS * len(PRESSURE_LEVELS)) + var_idx
                    ]
                    logs[
                        f"val_lw_rmse_{general_config.channel_names[var_idx + NUM_ATM_VARS]}"
                    ] = val_channel_lw_rmse[
                        (NUM_ATM_VARS * len(PRESSURE_LEVELS)) + var_idx
                    ]
                if args.encdec_cls == "dcae":
                    for var_idx in range(num_static_tensors):
                        logs[f"val_rmse_{general_config.static_names[var_idx]}"] = (
                            val_channel_rmse[
                                (NUM_ATM_VARS * len(PRESSURE_LEVELS))
                                + NUM_SUR_VARS
                                + var_idx
                            ]
                        )
                        logs[f"val_lw_rmse_{general_config.static_names[var_idx]}"] = (
                            val_channel_lw_rmse[
                                (NUM_ATM_VARS * len(PRESSURE_LEVELS))
                                + NUM_SUR_VARS
                                + var_idx
                            ]
                        )

                if args.aug_metric is not None:
                    logs["val_aug_loss"] = val_aug_loss
                accelerator.log(logs, step=global_step)

            if ema_config.use_ema:
                # Restore the encdec parameters.
                ema_model.restore(encdec.parameters())

    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Make sure to save and push the ema weight
        if ema_config.use_ema:
            ema_model.store(encdec.parameters())
            ema_model.copy_to(encdec.parameters())

        accelerator.unwrap_model(encdec).save_pretrained(
            os.path.join(general_config.output_dir, "encdec")
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=args.hub_model_id,
                folder_path=general_config.output_dir + "/encdec",
                path_in_repo=general_config.output_dir.split("/")[-1],
                commit_message="running weight",
                ignore_patterns=["checkpoint_"],
                token=args.hub_token if args.hub_token else None,
            )
        if ema_config.use_ema:
            ema_model.restore(encdec.parameters())
        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
