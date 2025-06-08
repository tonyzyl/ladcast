from typing import Optional

import torch
import torch.nn.functional as F


class LpLoss(object):
    # loss function with rel/abs Lp loss, modified from neuralop:
    # https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/data_losses.py
    """
    LpLoss: Lp loss function, return the relative loss by default
    Args:
        d: int, start dimension of the field. E,g., for shape like (b, c, h, w), d=2 (default 1)
        p: int, p in Lp norm, default 2
        reduce_dims: int or list of int, dimensions to reduce
        reductions: str or list of str, 'sum' or 'mean'

    Call: (y_pred, y)
    """

    def __init__(self, d=1, p=2, reduce_dims=0, reductions="sum"):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x, y, weight: Optional[torch.Tensor] = None):
        if weight is None:
            diff = torch.norm(
                torch.flatten(x, start_dim=-self.d)
                - torch.flatten(y, start_dim=-self.d),
                p=self.p,
                dim=-1,
                keepdim=False,
            )
        else:
            # weight: (B, C, H, 1)
            diff = torch.norm(
                (weight * (x - y)).flatten(start_dim=-self.d),
                p=self.p,
                dim=-1,
                keepdim=False,
            )

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y, weight: Optional[torch.Tensor] = None):
        if weight is None:
            diff = torch.norm(
                torch.flatten(x, start_dim=-self.d)
                - torch.flatten(y, start_dim=-self.d),
                p=self.p,
                dim=-1,
                keepdim=False,
            )
            ynorm = torch.norm(
                torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
            )
        else:
            # weight: (B, C, H, 1)
            diff = torch.norm(
                (weight * (x - y)).flatten(start_dim=-self.d),
                p=self.p,
                dim=-1,
                keepdim=False,
            )
            ynorm = torch.norm(
                (weight * y).flatten(start_dim=-self.d), p=self.p, dim=-1, keepdim=False
            )

        diff = diff / ynorm  # (B, C)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, weight: Optional[torch.Tensor] = None):
        return self.rel(y_pred, y, weight=weight)

    @torch.no_grad()
    def get_loss_per_var(
        self,
        y_pred,
        y,
        num_atm_vars,
        num_levels=13,
        weight: Optional[torch.Tensor] = None,
    ):
        """Assuming input of order [atm_vars, sur_vars]"""
        if weight is None:
            diff = torch.norm(
                torch.flatten(y_pred, start_dim=-self.d)
                - torch.flatten(y, start_dim=-self.d),
                p=self.p,
                dim=-1,
                keepdim=False,
            )
            ynorm = torch.norm(
                torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
            )
        else:
            # weight: (B, C, H, 1)
            diff = torch.norm(
                (weight * (y_pred - y)).flatten(start_dim=-self.d),
                p=self.p,
                dim=-1,
                keepdim=False,
            )
            ynorm = torch.norm(
                (weight * y).flatten(start_dim=-self.d), p=self.p, dim=-1, keepdim=False
            )

        diff = diff / ynorm  # (B, C)

        atm_channel_cutoff = num_atm_vars * num_levels
        batch_mean = torch.empty(
            y_pred.shape[1] - atm_channel_cutoff + num_atm_vars,
            device=y_pred.device,
            dtype=y_pred.dtype,
        )
        for i in range(num_atm_vars):
            batch_mean[i] = diff[:, i * num_levels : (i + 1) * num_levels].mean()
            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")
        for i in range(0, y_pred.shape[1] - atm_channel_cutoff):
            batch_mean[i + num_atm_vars] = diff[:, i].mean()
            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")

        return batch_mean  # (num_atm_vars + num_sur_vars,)


class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, y_pred, y):
        return F.mse_loss(y_pred.float(), y.float(), reduction=self.reduction)

    @torch.no_grad()
    def get_loss_per_var(
        self,
        y_pred,
        y,
        num_atm_vars,
        num_levels=13,
        weight: Optional[torch.Tensor] = None,
    ):
        """Assuming input of order [atm_vars, sur_vars]"""
        if weight is None:
            diff = (y_pred - y) ** 2
        else:
            diff = weight * ((y_pred - y) ** 2)
        atm_channel_cutoff = num_atm_vars * num_levels
        batch_mean = torch.empty(
            y_pred.shape[1] - atm_channel_cutoff + num_atm_vars,
            device=y_pred.device,
            dtype=y_pred.dtype,
        )
        for i in range(num_atm_vars):
            batch_mean[i] = diff[:, i * num_levels : (i + 1) * num_levels].mean()
            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")
        for i in range(0, y_pred.shape[1] - atm_channel_cutoff):
            batch_mean[i + num_atm_vars] = diff[:, i].mean()
            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")

        return batch_mean  # (num_atm_vars + num_sur_vars,)
