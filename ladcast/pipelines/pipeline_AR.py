from typing import List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from ladcast.pipelines.utils import Fields2DPipelineOutput


class AutoRegressive2DPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        ar_model
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self, ar_model, scheduler, scheduler_step_kwargs: Optional[dict] = None
    ):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        # scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(ar_model=ar_model, scheduler=scheduler)
        self.scheduler_step_kwargs = scheduler_step_kwargs or {}

    # copy of function call but return the trajectory
    def return_trajectory(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        return_dict: bool = True,
        known_latents: torch.Tensor = None,
        do_edm_style: bool = True,
        save_step: List[int] = None,
    ) -> Union[Fields2DPipelineOutput, Tuple]:
        raise NotImplementedError("This function is not implemented yet.")

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        return_seq_len: int = 1,
        known_latents: torch.Tensor = None,  # (B, C, H, W) or (B, C, T, H, W)
        timestamps: Optional[torch.LongTensor] = None,  # int format YYYYMMDDHH
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        return_dict: bool = True,
        do_edm_style: bool = True,
    ) -> Union[Fields2DPipelineOutput, Tuple]:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        assert known_latents is not None, "known_latents must be provided"

        image_shape = (
            batch_size,
            self.ar_model.config.out_channels,
            return_seq_len,
            *known_latents.shape[-2:],
        )

        image = randn_tensor(
            image_shape,
            generator=generator,
            device=self._execution_device,
            dtype=self.ar_model.dtype,
        )
        known_latents.to(self._execution_device)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            if do_edm_style:
                x_in = self.scheduler.scale_model_input(image, t)
                # x_in = torch.cat((x_in, concat_mask), dim=1)
                t = t.expand(batch_size).to(self._execution_device)
                model_output = self.ar_model(
                    x_in, t, known_latents, time_elapsed=timestamps, return_dict=False
                )[0]
            else:
                raise NotImplementedError("Only EDM style is supported for now")

            # 2. do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, **self.scheduler_step_kwargs, return_dict=False
            )[0]

        if not return_dict:
            return (image,)

        return Fields2DPipelineOutput(fields=image)
