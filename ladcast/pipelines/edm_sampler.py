from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

# from tqdm.auto import tqdm


@torch.no_grad()
def edm_AR_sampler(
    net,
    noise_scheduler,
    batch_size=1,
    return_seq_len=1,
    randn_like=torch.randn_like,
    num_inference_steps=18,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=0,
    deterministic=True,
    known_latents=None,
    timestamps: Optional[torch.LongTensor] = None,  # int format YYYYMMDDHH
    # return_trajectory=False,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    device="cpu",
):
    """
    known_latents: torch.Tensor, shape (B, C, T, H, W) or (B, C, H, W), initial solution profile, if
    T is provided, the 1st frame will be used as the initial condition
    """

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    assert known_latents is not None, "known_latents must be provided"

    if isinstance(device, str):
        device = torch.device(device)

    latents_shape = list(known_latents.shape)
    latents_shape = (
        batch_size,
        net.config.out_channels,
        return_seq_len,
        *latents_shape[-2:],
    )

    latents = randn_tensor(
        latents_shape, generator=generator, device=device, dtype=net.dtype
    )  # (B, C, T, H, W)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    t_steps = noise_scheduler.sigmas.to(device)

    x_next = latents.to(torch.float64) * t_steps[0]

    # if return_trajectory:
    #    whole_trajectory = torch.zeros((num_inference_steps, *x_next.shape), dtype=torch.float64)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if not deterministic:
            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_inference_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        tmp_x_hat = x_hat.clone()
        c_noise = noise_scheduler.precondition_noise(t_hat)
        # Euler step.
        tmp_x_hat = noise_scheduler.precondition_inputs(tmp_x_hat, t_hat)

        denoised = net(
            tmp_x_hat.to(torch.float32),
            c_noise.reshape(-1).to(torch.float32),
            known_latents,
            time_elapsed=timestamps,
        ).sample.to(torch.float64)
        denoised = noise_scheduler.precondition_outputs(x_hat, denoised, t_hat)

        d_cur = (
            x_hat - denoised
        ) / t_hat  # denoise has the same shape as x_hat (b, out_channels, h, w)
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_inference_steps - 1:
            tmp_x_next = x_next.clone()
            c_noise = noise_scheduler.precondition_noise(t_next)
            tmp_x_next = noise_scheduler.precondition_inputs(tmp_x_next, t_next)

            denoised = net(
                tmp_x_next.to(torch.float32),
                c_noise.reshape(-1).to(torch.float32),
                known_latents,
                time_elapsed=timestamps,
            ).sample.to(torch.float64)
            denoised = noise_scheduler.precondition_outputs(x_next, denoised, t_next)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # if return_trajectory:
        #    whole_trajectory[i] = x_next

    # if return_trajectory:
    #    return x_next, whole_trajectory
    return x_next.float()
