import numpy as np

import torch
import torch.nn as nn


def qr(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert inputs.ndim == 3 and inputs.shape[1] == inputs.shape[2]

    # NOTE: the current implementation of torch.linalg.qr can be numerically
    # unstable during backwards pass when output has (close to) linearly
    # dependent columns, at least until pivoting is implemented upstream
    # (see comment torch.linalg.qr docs, as well as
    # https://github.com/pytorch/pytorch/issues/42792). Hence we convert to
    # double before applying the QR (and then convert back)
    #
    # NOTE: In addition, for some reason, QR decomposition on GPU is very
    # slow in PyTorch. This is a known issue: see
    # https://github.com/pytorch/pytorch/issues/22573). We work around this
    # as follows, although this is *still* much slower than it could be if
    # it were properly taking advantage of the GPU...
    #
    Q, R = torch.linalg.qr(inputs.cpu().double())
    Q = Q.to(torch.get_default_dtype()).to(inputs.device)
    R = R.to(torch.get_default_dtype()).to(inputs.device)

    # This makes sure the diagonal is positive, so that the Q matrix is
    # unique (and coresponds to the output produced by Gram-Schmidt, which
    # is equivariant)
    diag_sgns = torch.diag_embed(torch.diagonal(R, dim1=-2, dim2=-1).sign())

    # *Shouldn't* do anything but just to be safe:
    diag_sgns = diag_sgns.detach()

    return Q @ diag_sgns, diag_sgns @ R


def gram(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)  # TODO: Test without
    return x @ x.transpose(1, 2)


def cholesky(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert x.shape[1] == x.shape[2]
    result = torch.linalg.cholesky(x)
    return result.to(torch.get_default_dtype())  # TODO: Test without


# TODO: Just make this take 1 argument
def flatten(*args: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.flatten(start_dim=1) for x in args], dim=1)


def make_square(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2

    d = np.sqrt(x.shape[1])

    assert d == int(d), "Input must be squareable"

    return x.reshape(x.shape[0], int(d), int(d))


def append(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert len(y.shape) == 2
    return torch.cat((x, y.unsqueeze(2)), dim=2)


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[1]
    return torch.bmm(x, y)


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    return x.transpose(1, 2)


def orthogonal_haar(dim: int, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Implements the method of https://arxiv.org/pdf/math-ph/0609050v2.pdf
    (see (5.12) of that paper in particular)
    """

    noise = torch.randn(target_tensor.shape[0], dim, dim, 
                        device=target_tensor.device)
    return qr(noise)[0]

def compute_gradient_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm(p=2.0)
    return norm


"""NOTE: from https://github.com/lsj2408/Transformer-M/tree/main"""

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, node_mask):
        # x: [bs, n_nodes, 3]
        # node_mask: [bs, n_nodes, 1]
        N = torch.sum(node_mask, dim=1, keepdim=True)  # [bs, 1, 1]
        dist_mat = torch.cdist(x, x).unsqueeze(-1)  # [bs, n_nodes, n_nodes]
        pos_emb = dist_mat.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        pos_emb = gaussian(pos_emb.float(), mean, std).type_as(self.means.weight)  # [bs, n_nodes, n_nodes, K]
        pos_emb = pos_emb * node_mask[:, :, :, None] * node_mask[:, None, :, :]
        return pos_emb
