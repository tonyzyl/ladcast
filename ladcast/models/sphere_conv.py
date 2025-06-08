from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class SphereConv2d(nn.Conv2d):
    """
    2D Convolution with circular padding in horizontal direction,
    and inverted reflection in vertical direction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode=None,
        padding_value=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )

        # assert padding == 1, "For now, SphereConv2d only tested on padding=1 for spherical convolution."
        # self.padding = padding
        # assert self.kernel_size[0] == self.kernel_size[1] == 3, \
        # "SphereConv2d currently only tested on 3x3 kernels for spherical convolution."
        assert self.stride[0] == self.stride[1] == 1, (
            "SphereConv2d currently only tested on stride=1 for spherical convolution. "
        )

        if padding_mode is not None:
            Warning(
                f"This module has special padding modes, the passed {padding_mode} will be ignored."
            )
        if padding_value is not None:
            Warning(
                f"This module has special padding modes, the passed {padding_value} will be ignored."
            )

    @staticmethod
    def sphere_pad(input: torch.Tensor, padding: Tuple[int] = (1, 1)) -> torch.Tensor:
        """
        Pad a 4D tensor (batch, channels, height, width) for spherical convolution.
        Uses circular padding for longitude (width) and special pole handling for latitude (height).
        The top and bottom rows still need to be handled

        Args:
            input: Input tensor of shape (B, C, H, W)
            padding: Number of padding elements on each side (padH, padW).

        Returns:
            Padded tensor with spherical boundary conditions
        """
        assert input.dim() == 4, (
            "Input tensor must be 4D (batch, channels, height, width)"
        )
        assert input.shape[3] % 2 == 0, (
            "Width of the input tensor must be even for proper shperical padding"
        )
        half_width = input.shape[3] // 2

        top_rows = input[:, :, : padding[0], :]
        top_rows = torch.roll(top_rows, shifts=half_width, dims=3)
        top_rows = torch.flip(top_rows, dims=[2])
        bottom_rows = input[:, :, -padding[0] :, :]
        bottom_rows = torch.roll(bottom_rows, shifts=half_width, dims=3)
        bottom_rows = torch.flip(bottom_rows, dims=[2])
        input = torch.cat([top_rows, input, bottom_rows], dim=2)

        return F.pad(input, (padding[1], padding[1], 0, 0), mode="circular")

    def top_conv(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to the top slice of the input tensor.
        This is used to handle the top row of the spherical convolution after padding.
        """
        # Flip the top row weight for top slice convolution
        kernel = self.weight.data
        kernel[:, :, : self.padding[0], :] = torch.flip(
            kernel[:, :, : self.padding[0], :], dims=[3]
        )
        output = F.conv2d(
            input, kernel, self.bias, self.stride, 0, self.dilation, self.groups
        )
        kernel[:, :, : self.padding[0], :] = torch.flip(
            kernel[:, :, : self.padding[0], :], dims=[3]
        )

        return output

    def bottom_conv(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to the bottom slice of the input tensor.
        This is used to handle the bottom row of the spherical convolution after padding.
        """
        # Flip the bottom row weight for bottom slice convolution
        kernel = self.weight.data
        kernel[:, :, -self.padding[0] :, :] = torch.flip(
            kernel[:, :, -self.padding[0] :, :], dims=[3]
        )
        output = F.conv2d(
            input, kernel, self.bias, self.stride, 0, self.dilation, self.groups
        )
        kernel[:, :, -self.padding[0] :, :] = torch.flip(
            kernel[:, :, -self.padding[0] :, :], dims=[3]
        )

        return output

    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        raise NotImplementedError(
            " SphereConv2d does not support _conv_forward method. Use forward method instead."
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (B, C, H, W)
        example:
        tmp = torch.arange(0, 24).view(1, 1, 3, 8)
        conv_cls = SphereConv2d(1, 1, 5, 1, 5//2)
        print(tmp)
        print(conv_cls.sphere_pad(tmp, (5//2, 5//2)))
        >>>
        tensor([[[[ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23]]]])
        tensor([[[[10, 11, 12, 13, 14, 15,  8,  9, 10, 11, 12, 13],
                [ 2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5],
                [ 6,  7,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1],
                [14, 15,  8,  9, 10, 11, 12, 13, 14, 15,  8,  9],
                [22, 23, 16, 17, 18, 19, 20, 21, 22, 23, 16, 17],
                [18, 19, 20, 21, 22, 23, 16, 17, 18, 19, 20, 21],
                [10, 11, 12, 13, 14, 15,  8,  9, 10, 11, 12, 13]]]])

        conv_cls.weight.data = torch.tensor([[[[0,1,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0]]]], requires_grad=True, dtype=torch.float32)
        conv_cls.bias.data = torch.tensor([0.0], requires_grad=True, dtype=torch.float32)
        print(conv_cls.weight.data.shape)
        >>>
        tensor([[[[0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 1., 0.]]]])

        conv_cls(tmp.float())
        >>>
        tensor([[[[44., 48., 52., 40., 44., 48., 52., 40.],
                [48., 44., 48., 44., 48., 44., 48., 44.],
                [52., 40., 44., 48., 52., 40., 44., 48.]]]], grad_fn=<CatBackward0>)
        """
        input = self.sphere_pad(input, padding=self.padding)
        top_slice = input[:, :, : self.kernel_size[0], :]
        mid_slice = input[:, :, self.stride[0] : -self.stride[0], :]
        bottom_slice = input[:, :, -self.kernel_size[0] :, :]
        top_slice = self.top_conv(top_slice)
        # print("top slice", top_slice, top_slice.shape)
        mid_slice = F.conv2d(
            mid_slice,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
        # print("mid slice", mid_slice, mid_slice.shape)
        bottom_slice = self.bottom_conv(bottom_slice)
        # print("bottom slice", bottom_slice, bottom_slice.shape)
        return torch.cat([top_slice, mid_slice, bottom_slice], dim=2)
