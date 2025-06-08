from typing import Tuple, Union

import torch


class Karras_sigmas_lognormal:
    def __init__(
        self,
        sigmas,
        P_mean_start=-1.2,
        P_std_start=1.2,
        P_mean_end=1.2,
        P_std_end=1.7,
        num_max_steps=50000,
    ):
        self.P_means = torch.linspace(P_mean_start, P_mean_end, num_max_steps)
        self.P_stds = torch.linspace(P_std_start, P_std_end, num_max_steps)
        self.num_max_steps = num_max_steps
        self.sigmas = sigmas
        self.P_mean_start = P_mean_start
        self.P_std_start = P_std_start
        self.P_mean_end = P_mean_end
        self.P_std_end = P_std_end

    def __call__(self, batch_size, cur_step, generator=None, device="cpu"):
        rnd_normal = torch.randn(
            [batch_size, 1, 1, 1], device=device, generator=generator
        )
        step = min(cur_step, self.num_max_steps - 1)
        sigma = (
            rnd_normal * self.P_stds[step].to(device) + self.P_means[step].to(device)
        ).exp()
        # if cur_step <= self.num_max_steps:
        # sigma = (rnd_normal * self.P_std_start + self.P_mean_start).exp()
        # else:
        # sigma = (rnd_normal * self.P_std_end + self.P_mean_end).exp()
        # Find the indices of the closest matches
        # Expand self.sigmas to match the batch size
        # sigmas get concatenated with 0 at the end
        sigmas_expanded = self.sigmas[:-1].view(1, -1).to(device)
        sigma_expanded = sigma.view(batch_size, 1)

        # Calculate the difference and find the indices of the minimum difference
        diff = torch.abs(sigmas_expanded - sigma_expanded)
        indices = torch.argmin(diff, dim=1)

        return indices


def apply_batch_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modified diffusers.models.embeddings.apply_rotary_emb
    Args:
        freqs_cis: Tuple of two tensors, each of shape [B, S, D]
    """
    if use_real:
        cos, sin = freqs_cis  # [B, S, D]
        cos = cos.unsqueeze(1)  # prev: cos = cos[None, None]
        sin = sin.unsqueeze(1)
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"Invalid use_real_unbind_dim: {use_real_unbind_dim}")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        raise NotImplementedError
