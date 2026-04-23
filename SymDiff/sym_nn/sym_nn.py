import torch
import torch.nn as nn

from equivariant_diffusion.utils import remove_mean_with_mask, assert_correctly_masked
from sym_nn.utils import qr, orthogonal_haar, GaussianLayer
from sym_nn.dit import DiT

from timm.models.vision_transformer import Mlp


class DiTGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        mlp_type: str = "swiglu",

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims

        self.gaussian_embedder = GaussianLayer(K=K)

        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        self.model = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads, 
            mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout, mlp_type=mlp_type,
            use_fused_attn=True, x_emb="identity")

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        """
        Computes a forward pass of \epsilon_\theta.

        Args: 
            t: [bs, 1]
            xh: [bs, n_nodes, n_dims+in_nodes_nf]
            node_mask: [bs, n_nodes, 1]
        Returns:
            xh: [bs, n_nodes, n_dims+in_nodes_nf]
        """

        assert context is None

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        xh = torch.cat([x.clone(), h], dim=-1)
        xh = self.xh_embedder(xh)

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, K]

        xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
        xh = node_mask * self.model(xh, t.squeeze(-1), node_mask.squeeze(-1))

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)

        return xh


class DiT_DitGaussian_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        xh_hidden_size: int,
        K: int,

        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,

        dec_hidden_features: int,

        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_dropout: float,

        noise_dims: int = 0,
        noise_std: float = 1.0,

        mlp_type: str = "swiglu",

        n_dims: int = 3,
        device: str = "cpu",

        fix_qr: bool = True
    ) -> None:
        super().__init__()

        self.args = args
        self.n_dims = n_dims
        self.fix_qr = fix_qr

        self.gaussian_embedder = GaussianLayer(K=K)            

        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        self.noise_dims = noise_dims
        self.noise_std = noise_std

        self.gamma_enc_input_dim = n_dims + hidden_size - xh_hidden_size + noise_dims

        self.gamma_enc = DiT(
            out_channels=0, 
            hidden_size=enc_hidden_size, depth=enc_depth, num_heads=enc_num_heads, 
            mlp_ratio=enc_mlp_ratio, mlp_dropout=mlp_dropout, mlp_type=mlp_type,
            use_fused_attn=True, x_emb="linear", 
            input_dim=self.gamma_enc_input_dim).to(device)

        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, 
            hidden_features=dec_hidden_features,
            out_features=n_dims**2).to(device)

        self.k = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf, 
            hidden_size=hidden_size, depth=depth, num_heads=num_heads, 
            mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout, mlp_type=mlp_type,
            use_fused_attn=True, x_emb="identity").to(device)

        self.device = device

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)
        self.gamma_dec.apply(_basic_init)

    def forward(self):
        raise NotImplementedError

    def base_gamma(self, t, x):
        """Samples from the base gamma kernel - i.e. the Haar."""
        return orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]

    def gamma_backbone(self, t, x, node_mask):
        """Samples from the backbone of gamma - i.e. f_\theta."""
        bs, n_nodes, _ = x.shape

        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb_gamma = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb_gamma = torch.sum(self.pos_embedder(pos_emb_gamma), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        if self.noise_dims > 0:
            x = torch.cat([
                x, 
                node_mask * self.noise_std * torch.randn(
                    bs, n_nodes, self.noise_dims, device=self.device)
                ], dim=-1)

        x = torch.cat([x, pos_emb_gamma], dim=-1)

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False)

        # Decoded summed representation into gamma - this is S_n-invariant
        gamma = torch.sum(gamma, dim=1) / N.squeeze(-1)  # [bs, hidden_size]

        # [bs, 3, 3] - correct direction of QR for right equivariance
        if self.fix_qr:
            gamma = qr(
                self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims).transpose(1, 2)
                )[0].transpose(1, 2)
        else:
            gamma = qr(
                self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims) 
                )[0]

        return gamma

    def gamma(self, t, x, node_mask):
        """Samples from the overall gamma kernel."""
        base_gamma = self.base_gamma(t, x)
        g_inv_x = torch.bmm(x.clone(), base_gamma.clone())  # as x is represented row-wise
        gamma = self.gamma_backbone(t, g_inv_x, node_mask)
        gamma = torch.bmm(gamma, base_gamma.transpose(1, 2))
        return gamma

    def k_backbone(self, t, x, h, node_mask):
        """Computes the backbone of the kernel - i.e. \epsilon_\theta."""
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb_k = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb_k = torch.sum(self.pos_embedder(pos_emb_k), dim=-2) / N  # [bs, n_nodes, hidden_size-xh_hidden_size]

        xh = self.xh_embedder(torch.cat([x, h], dim=-1))
        xh = node_mask * torch.cat([xh, pos_emb_k], dim=-1)
        xh = node_mask * self.k(xh, t.squeeze(-1), node_mask.squeeze(-1))

        return xh

    def _forward(self, t, xh, node_mask, edge_mask, context):
        """Samples from the overall symmetrised kernel.

        Args: 
            t: [bs, 1]
            xh: [bs, n_nodes, n_dims+in_nodes_nf]
            node_mask: [bs, n_nodes, 1]
        Returns:
            xh: [bs, n_nodes, n_dims+in_nodes_nf]
        """

        assert context is None

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        gamma = self.gamma(t, x, node_mask)
        g_inv_x = torch.bmm(x, gamma)
        xh = self.k_backbone(t, g_inv_x, h, node_mask)

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        x = torch.bmm(x, gamma.transpose(1, 2))

        xh = torch.cat([x, h], dim=-1)
        assert_correctly_masked(xh, node_mask)

        return xh


if __name__ == "__main__":
    pass