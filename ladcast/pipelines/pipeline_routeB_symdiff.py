from __future__ import annotations

import math

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

from ladcast.models.routeB_fourier_shift import (
    apply_fourier_shift_5d,
    invert_fourier_shift_5d,
)
from ladcast.models.symmetry import sample_lon_roll_shifts


class RouteBSymDiffPipeline:
    """Sampling pipeline supporting both roll-based and Fourier-based symmetrisation."""

    def __init__(self, denoiser, scheduler: DDPMScheduler):
        self.denoiser = denoiser
        self.scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    @property
    def symmetry_type(self) -> str:
        return getattr(self.denoiser, "symmetry_type", "roll")

    def to(self, device: torch.device | str):
        self.denoiser = self.denoiser.to(device)
        return self

    # ---------------------------------------------------------------
    # Roll-based group state (legacy)
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Fourier-based group state
    # ---------------------------------------------------------------
    def _fourier_group_state(
            self,
            batch_size: int,
            device: torch.device,
            inference_mode: str,
            fixed_phase: float,
    ) -> torch.Tensor | None:
        if inference_mode == "fourier_identity":
            return None
        if inference_mode == "fourier_random_single":
            return torch.rand(batch_size, device=device) * 2 * math.pi
        if inference_mode == "fourier_fixed":
            return torch.full((batch_size,), float(fixed_phase), device=device)
        raise ValueError(f"Unsupported fourier inference_mode: {inference_mode}")

    # ---------------------------------------------------------------
    # Single-sample generation
    # ---------------------------------------------------------------
    @torch.no_grad()
    def sample_once(
            self,
            cond: torch.Tensor,
            num_inference_steps: int = 50,
            generator: torch.Generator | None = None,
            max_lon_shift: int = 16,
            inference_mode: str = "random_single",
            fixed_shift: int = 0,
            fixed_phase: float = 0.0,
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

        is_fourier = inference_mode.startswith("fourier_")

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            t_batch = t.expand(b)
            if is_fourier:
                shifts = self._fourier_group_state(b, device, inference_mode, fixed_phase)
            else:
                shifts = self._group_state_for_step(b, device, inference_mode, max_lon_shift, fixed_shift)
            pred_eps = self.denoiser(cond, sample, t_batch, group_state=shifts)
            sample = self.scheduler.step(pred_eps, t, sample).prev_sample
        return sample

    # ---------------------------------------------------------------
    # Fourier grid sampling (uniform phase grid, shared initial noise)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def sample_fourier_grid(
            self,
            cond: torch.Tensor,
            num_inference_steps: int = 50,
            n_grid: int = 8,
            generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample on a uniform phase grid and average results."""
        device = self.device
        cond = cond.to(device)
        b, _, c, h, w = cond.shape
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / n_grid), n_grid, device=device)

        accumulated = torch.zeros(b, self.denoiser.target_seq_len, c, h, w, device=device)

        for phi_val in angles:
            phi = phi_val.expand(b)
            # Shift condition
            cond_shifted = apply_fourier_shift_5d(cond, -phi)
            # Generate from shifted condition
            sample = randn_tensor(
                (b, self.denoiser.target_seq_len, c, h, w),
                generator=generator,
                device=device,
                dtype=cond.dtype,
            )
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for t in self.scheduler.timesteps:
                t_batch = t.expand(b)
                pred_eps = self.denoiser(cond_shifted, sample, t_batch, group_state=None)
                sample = self.scheduler.step(pred_eps, t, sample).prev_sample
            # Un-shift result
            accumulated += invert_fourier_shift_5d(sample, -phi)

        return accumulated / n_grid

    # ---------------------------------------------------------------
    # Main entrypoint
    # ---------------------------------------------------------------
    @torch.no_grad()
    def __call__(
            self,
            cond: torch.Tensor,
            num_inference_steps: int = 50,
            generator: torch.Generator | None = None,
            max_lon_shift: int = 16,
            inference_mode: str = "random_single",
            fixed_shift: int = 0,
            fixed_phase: float = 0.0,
            num_symmetry_samples: int = 1,
    ) -> torch.Tensor:
        # --- Single-sample modes (roll + fourier) ---
        single_modes = {
            "identity", "random_single", "fixed_group",
            "fourier_identity", "fourier_random_single", "fourier_fixed",
        }
        if inference_mode in single_modes:
            return self.sample_once(
                cond,
                num_inference_steps=num_inference_steps,
                generator=generator,
                max_lon_shift=max_lon_shift,
                inference_mode=inference_mode,
                fixed_shift=fixed_shift,
                fixed_phase=fixed_phase,
            )

        # --- Roll group_mean ---
        if inference_mode == "group_mean":
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

        # --- Fourier mean (random phases, average) ---
        if inference_mode == "fourier_mean":
            if num_symmetry_samples <= 0:
                raise ValueError("num_symmetry_samples must be > 0 for fourier_mean")
            samples = [
                self.sample_once(
                    cond,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    max_lon_shift=max_lon_shift,
                    inference_mode="fourier_random_single",
                )
                for _ in range(num_symmetry_samples)
            ]
            return torch.stack(samples, dim=0).mean(dim=0)

        # --- Fourier grid (uniform phase grid, average) ---
        if inference_mode == "fourier_grid":
            return self.sample_fourier_grid(
                cond,
                num_inference_steps=num_inference_steps,
                n_grid=max(num_symmetry_samples, 2),
                generator=generator,
            )

        raise ValueError(f"Unsupported inference_mode: {inference_mode}")
