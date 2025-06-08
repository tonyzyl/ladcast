# Copyright 2025 Yilin Zhuang
# Based on work by The Hunyuan Team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified based on the HunyuanVideo
# https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/transformers/transformer_hunyuan_video.py
# Please refer to the annotations under each modified class or function for more details.

import warnings
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    TimestepEmbedding,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph

from ladcast.evaluate.utils import get_normalized_lat_weights_based_on_cos
from ladcast.models.embeddings import (
    HunyuanVideoPatchEmbed,
    LaDCastRotaryPosEmbed_from_grid,
    get_year_sincos_embedding,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LaDCastAttnProcessor2_0:
    """
    Modified from HunyuanVideoAttnProcessor2_0
    # TODO:
    #Changes: encoder_hidden_states is image-like and shall have the same resolution
    #         -> apply conditioning rotary embeddings to the encoder_hidden_states as well.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LaDCastAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,  # shape: [B, N, C]
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            # add_q_proj is None in single-stream
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(
            1, 2
        )  # shape: [B, heads, N, head_dim]
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                # add_q_proj is None in single-stream
                # print('1st part: ', query[:, :, : -encoder_hidden_states.shape[1]].shape, ' 2nd part: ', query[:, :, -encoder_hidden_states.shape[1] :].shape)
                if cond_image_rotary_emb is not None:
                    query = torch.cat(
                        [
                            apply_rotary_emb(
                                query[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            apply_rotary_emb(
                                query[:, :, -encoder_hidden_states.shape[1] :],
                                cond_image_rotary_emb,
                            ),
                        ],
                        dim=2,
                    )
                    key = torch.cat(
                        [
                            apply_rotary_emb(
                                key[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            apply_rotary_emb(
                                key[:, :, -encoder_hidden_states.shape[1] :],
                                cond_image_rotary_emb,
                            ),
                        ],
                        dim=2,
                    )
                else:
                    warnings.warn(
                        "no embedding for conditioning tensor triggered, this is not expected, please check the code"
                    )
                    query = torch.cat(
                        [
                            apply_rotary_emb(
                                query[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            query[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                    key = torch.cat(
                        [
                            apply_rotary_emb(
                                key[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            key[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
            else:
                # called in refiner & dual-stream (might change)
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            # TODO: modify encoder_hidden_states emb
            # add_q_proj is not None in dual-stream
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        """
        attn_weight = query @ key.transpose(-2, -1) * scale_factor # [B, heads, N_seq, N_dim] @ [B, heads, N_dim, N_seq] -> [B, heads, N_seq, N_seq]
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        """
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)  # -> [B, N, C]
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                # True in dual-streams
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class LaDCastIndividualTokenRefinerBlock(nn.Module):
    """Modified from HunyuanVideoIndividualTokenRefinerBlock"""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-7)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            processor=LaDCastAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-7,
            pre_only=True,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-7)
        self.ff = FeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
            dropout=mlp_drop_rate,
        )

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output, _ = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class LaDCastIndividualTokenRefiner(nn.Module):
    """modified from HunyuanVideoIndividualTokenRefiner"""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                LaDCastIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, attention_mask, image_rotary_emb)

        return hidden_states


class LaDCastTokenRefiner(nn.Module):
    """Modified from HunyuanVideoTokenRefiner"""

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = LaDCastIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        pooled_projections = hidden_states.mean(dim=1)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(
            hidden_states, temb, attention_mask, image_rotary_emb
        )

        return hidden_states


@maybe_allow_in_graph
class LaDCastSingleTransformerBlock(nn.Module):
    # Modified from HunyuanVideoSingleTransformerBlock
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=LaDCastAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-7,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            cond_image_rotary_emb=cond_image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class LaDCastTransformerBlock(nn.Module):
    # Modified from HunyuanVideoTransformerBlock
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=LaDCastAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-7,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-7)
        self.ff = FeedForward(
            hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

        self.norm2_context = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-7
        )
        self.ff_context = FeedForward(
            hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            cond_image_rotary_emb=cond_freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )

        return hidden_states, encoder_hidden_states


class LaDCastTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    r"""
    Modified from HunyuanTransformer3DModel:
    A Transformer model for video-like data used in [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).
    The original encoder_hidden_states already contains the positional information, if we take
    the conditioning tensor as input, we need to add rotary pos emb to that before refining.

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.
        num_refiner_layers (`int`, defaults to `2`):
            The number of layers of refiner blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings in the model.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        pooled_projection_dim (`int`, defaults to `768`):
            The dimension of the pooled projection of the text embeddings.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
    """

    _supports_gradient_checkpointing = True

    @staticmethod
    def _grid_deg2rad(grid_pos: Union[Number, List[Number], Tuple[Number]]):
        if isinstance(grid_pos, (list, tuple)):
            grid_pos = [np.deg2rad(x) for x in grid_pos]
        else:
            grid_pos = np.deg2rad(grid_pos)
        return grid_pos

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        rope_spatial_grid_start_pos: Union[Number, List[Number], Tuple[Number]] = 0,
        rope_spatial_grid_end_pos: Optional[
            Union[Number, List[Number], Tuple[Number]]
        ] = None,
        spatial_deg2rad: bool = False,
        conditioning_tensor_in_channels: int = None,
        conditioning_tensor_intermediate_proj_dim: Optional[int] = None,
        conditioning_tensor_rope_axes_dim: Tuple[int] = (16, 56, 56),
        incl_time_elapsed: bool = False,
        nope: bool = False,
        scale_attn_by_lat: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim
        )
        if conditioning_tensor_intermediate_proj_dim is None:
            conditioning_tensor_intermediate_proj_dim = inner_dim
        self.context_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size),
            conditioning_tensor_in_channels,
            inner_dim,
        )
        self.context_refiner = LaDCastTokenRefiner(
            conditioning_tensor_intermediate_proj_dim,
            num_attention_heads,
            attention_head_dim,
            num_layers=num_refiner_layers,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(inner_dim, inner_dim)

        if incl_time_elapsed:
            self.time_elapsed_embed = TimestepEmbedding(
                in_channels=256, time_embed_dim=2 * inner_dim
            )
        else:
            self.time_elapsed_embed = None

        self.scale_attn_by_lat = scale_attn_by_lat
        if scale_attn_by_lat:
            tmp_lat_weight = get_normalized_lat_weights_based_on_cos(
                np.linspace(-83.25, 84.75, 15)
            )  # (lat,)
            # normalize -> sum = 1
            tmp_lat_weight = torch.from_numpy(
                tmp_lat_weight / tmp_lat_weight.sum()
            ).float()
            # attn_weights (or attn_mask) requires broadcastable with [B, heads, N_seq, N_seq]
            tmp_lat_weight = tmp_lat_weight.repeat_interleave(30)  # (H*W,)
            self.attn_lat_weights = tmp_lat_weight.view(1, 1, 1, -1)

        # 2. RoPE
        if spatial_deg2rad:
            rope_spatial_grid_start_pos = self._grid_deg2rad(
                rope_spatial_grid_start_pos
            )
            rope_spatial_grid_end_pos = self._grid_deg2rad(rope_spatial_grid_end_pos)
        self.rope_spatial_grid_start_pos = rope_spatial_grid_start_pos
        self.rope_spatial_grid_end_pos = rope_spatial_grid_end_pos
        assert sum(rope_axes_dim) == attention_head_dim, (
            "sum(rope_axes_dim) must equal attention_head_dim"
        )

        assert sum(conditioning_tensor_rope_axes_dim) == attention_head_dim, (
            "sum(conditioning_tensor_rope_axes_dim) must equal attention_head_dim"
        )
        if nope:
            self.rope = None
            self.cond_rope = None
        else:
            self.rope = LaDCastRotaryPosEmbed_from_grid(
                rope_dim_list=rope_axes_dim,
                patch_size_list=[patch_size_t, patch_size, patch_size],
                theta=rope_theta,
            )
            self.cond_rope = LaDCastRotaryPosEmbed_from_grid(
                rope_dim_list=conditioning_tensor_rope_axes_dim,
                patch_size_list=[patch_size_t, patch_size, patch_size],
                theta=rope_theta,
            )

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                # HunyuanVideoTransformerBlock(
                LaDCastTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                # HunyuanVideoSingleTransformerBlock(
                LaDCastSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-7
        )
        self.proj_out = nn.Linear(
            inner_dim, patch_size_t * patch_size * patch_size * out_channels
        )

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        # encoder_hidden_states: torch.Tensor, # shape: [B, S, C]
        conditioning_tensors: torch.Tensor,
        # pooled_projections: torch.Tensor,
        # guidance: torch.Tensor = None,
        time_elapsed: Optional[torch.LongTensor] = None,  # [B] int of format YYYYMMDDHH
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        coords: Optional[
            torch.Tensor
        ] = None,  # (B, 2), containing (x, y) coordinates of the top left for each batch
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, return_seq_len, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_input_seq_len = conditioning_tensors.shape[2] // p_t
        post_patch_return_seq_len = return_seq_len // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        if self.scale_attn_by_lat:
            pred_attention_mask = self.attn_lat_weights.repeat(
                1, 1, 1, (post_patch_input_seq_len + post_patch_return_seq_len)
            ).to(hidden_states.device)
            cond_attention_mask = self.attn_lat_weights.repeat(
                1, 1, 1, post_patch_input_seq_len
            ).to(hidden_states.device)
        else:
            pred_attention_mask = None
            cond_attention_mask = None

        # 1. RoPE
        cond_temperal_coord = torch.arange(
            -post_patch_input_seq_len + 1,
            1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        pred_temperal_coord = torch.arange(
            1,
            post_patch_return_seq_len + 1,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if self.config.nope:
            pred_cos, pred_sin = get_1d_rotary_pos_embed(
                self.config.attention_head_dim,
                pred_temperal_coord,
                self.config.rope_theta,
                use_real=True,
            )
            image_rotary_emb = (
                pred_cos.repeat_interleave(post_patch_height * post_patch_width, dim=0),
                pred_sin.repeat_interleave(post_patch_height * post_patch_width, dim=0),
            )
            cond_cos, cond_sin = get_1d_rotary_pos_embed(
                self.config.attention_head_dim,
                cond_temperal_coord,
                self.config.rope_theta,
                use_real=True,
            )
            cond_image_rotary_emb = (
                cond_cos.repeat_interleave(post_patch_height * post_patch_width, dim=0),
                cond_sin.repeat_interleave(post_patch_height * post_patch_width, dim=0),
            )
        else:
            lat_coord = torch.linspace(
                self.rope_spatial_grid_start_pos[0],
                self.rope_spatial_grid_end_pos[0],
                steps=post_patch_height,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            lon_coord = torch.linspace(
                self.rope_spatial_grid_start_pos[1],
                self.rope_spatial_grid_end_pos[1],
                steps=post_patch_width,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            cond_grid_list = [cond_temperal_coord, lat_coord, lon_coord]
            pred_grid_list = [pred_temperal_coord, lat_coord, lon_coord]
            image_rotary_emb = self.rope(hidden_states, rope_grid_list=pred_grid_list)
            cond_image_rotary_emb = self.cond_rope(
                conditioning_tensors, rope_grid_list=cond_grid_list
            )

        # 2. Conditional embeddings
        # temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)  # -> [B, N_patch, C]
        encoder_hidden_states = self.context_embedder(
            conditioning_tensors
        )  # -> [B, N_patch, C]
        encoder_hidden_states = self.context_refiner(
            encoder_hidden_states,
            timestep,
            image_rotary_emb=cond_image_rotary_emb,
            attention_mask=cond_attention_mask,
        )

        with torch.autocast(hidden_states.device.type, torch.float32):
            temb = self.time_text_embed(
                timestep, encoder_hidden_states.mean(dim=1)
            )  # -> [B, C]

            if time_elapsed is not None:
                if self.time_elapsed_embed is not None:
                    time_elapsed_emb = get_year_sincos_embedding(
                        time_elapsed, embedding_dim=256
                    )
                    time_elapsed_emb = self.time_elapsed_embed(
                        time_elapsed_emb.to(hidden_states.device)
                    )
                    time_elapsed_scale, time_elapsed_shift = time_elapsed_emb.chunk(
                        2, dim=-1
                    )  # -> [B, C] each
                    temb = temb * (1 + time_elapsed_scale) + time_elapsed_shift
                else:
                    logger.warning(
                        "time_elapsed is provided, but time_elapsed_embed is not set. The model will not use time elapsed information."
                    )

        # 3. Attention mask preparation
        # Skipped, we are using initial profile as "text_embed", i.e., attention_mask = True for all patches

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        pred_attention_mask,
                        image_rotary_emb,
                        cond_image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        pred_attention_mask,
                        image_rotary_emb,
                        cond_image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    pred_attention_mask,
                    image_rotary_emb,
                    cond_image_rotary_emb,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    pred_attention_mask,
                    image_rotary_emb,
                    cond_image_rotary_emb,
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_return_seq_len,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(
            0, 4, 1, 5, 2, 6, 3, 7
        )  # (B, C, T, PS_t, Num_height, PS, Num_width, PS)
        hidden_states = (
            hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        )  # -> (B, C, T, H, W)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
