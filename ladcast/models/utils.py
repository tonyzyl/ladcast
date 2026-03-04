import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from diffusers.optimization import SchedulerType, get_scheduler

from typing import Tuple, Union


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


def get_scheduler_with_min_lr(
    name: str,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    min_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Wrapper around diffusers get_scheduler that enforces a minimum learning rate.
    
    Args:
        base_lr: The starting learning rate (must match optimizer's initial lr).
        min_lr: The target minimum learning rate.
    """
    min_lr_ratio = min_lr / base_lr
    if name == "polynomial":
        from diffusers.optimization import get_polynomial_decay_schedule_with_warmup
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=min_lr,  # Native support
            power=power,
            last_epoch=last_epoch
        )

    if name == "cosine":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step > num_training_steps:
                return min_lr_ratio

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress)) # Standard 1->0
            
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    if name == "cosine_with_restarts":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step > num_training_steps:
                return min_lr_ratio

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            
            if progress >= 1.0:
                return min_lr_ratio
            
            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    # 4. Fallback for others (Constant, etc. don't need min_lr usually)
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        power=power,
    )


def torch_gather(tensor, dim=0, stack=False):
    """
    Gathers a tensor from all distributed workers.
    
    Args:
        tensor: The tensor to gather.
        dim (int): Dimension to concatenate on if stack=False.
        stack (bool): If True, stacks results on new dim 0 (shape: [World, ...]).
                      If False, concatenates on dim 0 (shape: [World * Dim0, ...]).
    """
    # 1. Handle non-distributed case (e.g. debugging locally)
    if not torch.distributed.is_initialized():
        return tensor.unsqueeze(0) if stack else tensor
    
    # 2. Prepare list for all-gather
    world_size = torch.distributed.get_world_size()
    # We must ensure the tensor is contiguous for all_gather
    tensor = tensor.contiguous() 
    gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # 3. Collect (Synchronization Point - All workers must call this)
    torch.distributed.all_gather(gathered_list, tensor)
    
    # 4. Combine
    if stack:
        return torch.stack(gathered_list, dim=0)
    else:
        return torch.cat(gathered_list, dim=dim)