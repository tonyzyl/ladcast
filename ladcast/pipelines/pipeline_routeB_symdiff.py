from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

from ladcast.models.symmetry import sample_lon_roll_shifts


class RouteBSymDiffPipeline:
    def __init__(self, denoiser, scheduler: DDPMScheduler):
        self.denoiser = denoiser
        self.scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    def to(self, device: torch.device | str):
        self.denoiser = self.denoiser.to(device)
        return self

    def _group_state_for_step(
            self,
            batch_size: int,
            device: torch.device,
            inference_mode: str,
            max_lon_shift: int,
            fixed_shift: int,
    ) -> torch.Tensor | None:
        if inference_mode == "identity" or max_lon_shift <= 0:
            return None
        if inference_mode == "random_single":
            return sample_lon_roll_shifts(batch_size, max_lon_shift, device)
        if inference_mode == "fixed_group":
            return torch.full((batch_size,), int(fixed_shift), dtype=torch.long, device=device)
        raise ValueError(f"Unsupported single-sample inference_mode: {inference_mode}")

    @torch.no_grad()
    def sample_once(
            self,
            cond: torch.Tensor,
            num_inference_steps: int = 50,
            generator: torch.Generator | None = None,
            max_lon_shift: int = 16,
            inference_mode: str = "random_single",
            fixed_shift: int = 0,
    ) -> torch.Tensor:
        device = self.device
        cond = cond.to(device)
        b, _, c, h, w = cond.shape
        sample = randn_tensor(
            (b, self.denoiser.target_seq_len, c, h, w),
            generator=generator,
            device=device,
            dtype=cond.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            t_batch = t.expand(b)
            shifts = self._group_state_for_step(
                batch_size=b,
                device=device,
                inference_mode=inference_mode,
                max_lon_shift=max_lon_shift,
                fixed_shift=fixed_shift,
            )
            pred_eps = self.denoiser(cond, sample, t_batch, group_state=shifts)
            sample = self.scheduler.step(pred_eps, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def __call__(
            self,
            cond: torch.Tensor,
            num_inference_steps: int = 50,
            generator: torch.Generator | None = None,
            max_lon_shift: int = 16,
            inference_mode: str = "random_single",
            fixed_shift: int = 0,
            num_symmetry_samples: int = 1,
    ) -> torch.Tensor:
        if inference_mode in {"identity", "random_single", "fixed_group"}:
            return self.sample_once(
                cond,
                num_inference_steps=num_inference_steps,
                generator=generator,
                max_lon_shift=max_lon_shift,
                inference_mode=inference_mode,
                fixed_shift=fixed_shift,
            )
        if inference_mode != "group_mean":
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")
        if num_symmetry_samples <= 0:
            raise ValueError("num_symmetry_samples must be > 0 for group_mean")

        samples = [
            self.sample_once(
                cond,
                num_inference_steps=num_inference_steps,
                generator=generator,
                max_lon_shift=max_lon_shift,
                inference_mode="random_single",
                fixed_shift=fixed_shift,
            )
            for _ in range(num_symmetry_samples)
        ]
        return torch.stack(samples, dim=0).mean(dim=0)
