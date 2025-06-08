from typing import Optional, Tuple

import torch


def remove_channel(tensor: torch.Tensor, channel_idx: int) -> torch.Tensor:
    """
    Remove a channel from a tensor
    Args:
        tensor: (B, C, ...) tensor
        channel_idx: index of the channel to remove
    Returns:
        tensor with the specified channel removed
    """
    return torch.cat(
        [tensor[:, :channel_idx, ...], tensor[:, channel_idx + 1 :, ...]], dim=1
    )


def process_tensor_for_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    nan_mask: torch.Tensor,
    sst_chanel_idx: int,
    sur_pressure_channel_idx_to_remove: Optional[int] = None,
    nan_mask_val: int = -2.0,  # see dataloader.weather_dataset.weather_dataset_preprocess_batch
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The target has one more channel than the reconstructed tensor. The last channel is the surface pressure
    Args:
        reconstructed: (B, C, H, W) tensor
        target: (B, C+1, H, W) tensor
        nan_mask: (B, H, W) tensor
    """
    if sur_pressure_channel_idx_to_remove is not None:
        assert sst_chanel_idx < sur_pressure_channel_idx_to_remove, (
            "The sea surface temperature channel index must be less than the surface pressure channel index"
        )
        target = remove_channel(target, sur_pressure_channel_idx_to_remove)
    # Mask out NaN in sea surface temperature, avoid inplace operation
    reconstructed = torch.where(
        nan_mask.unsqueeze(1)
        & (
            torch.arange(reconstructed.size(1), device=reconstructed.device).view(
                1, -1, 1, 1
            )
            == sst_chanel_idx
        ),
        nan_mask_val,
        reconstructed,
    )

    target = torch.where(
        nan_mask.unsqueeze(1)
        & (
            torch.arange(target.size(1), device=target.device).view(1, -1, 1, 1)
            == sst_chanel_idx
        ),
        nan_mask_val,
        target,
    )
    # remove the surface pressure in the target
    return reconstructed, target
