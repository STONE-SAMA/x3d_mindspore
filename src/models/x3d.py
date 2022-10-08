# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" X3D network."""

import math
from typing import Optional, Tuple, Type

import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, HeNormal, Zero

from src.models.layers.swish import Swish
from src.models.layers.dropout_dense import DropoutDense
from src.models.layers.avgpool3d import AvgPool3D
from src.models.layers.squeeze_excite3d import SqueezeExcite3D
from src.models.layers.resnet3d import ResNet3D, ResidualBlock3D
from src.models.layers.inflate_conv3d import Inflate3D
from src.models.layers.unit3d import Unit3D
from src.utils.class_factory import ClassFactory, ModuleType


__all__ = [
    'x3d_xs',
    'x3d_s',
    'x3d_m',
    'x3d_l'
]


def drop_path(x: Tensor,
              drop_prob: float = 0.0,
              training: bool = False
              ):
    """Stochastic Depth per sample."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    uniform_real_op = ops.UniformReal()
    mask = keep_prob + uniform_real_op(shape)
    floor = ops.Floor()
    mask = floor(mask)
    div = ops.Div()
    return div(x, keep_prob) * mask


class BlockX3D(ResidualBlock3D):
    """
    BlockX3D 3d building block for X3D.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        conv12(nn.Cell, optional): Block that constructs first two conv layers.
            It can be `Inflate3D`, `Conv2Plus1D` or other custom blocks, this
            block should construct a layer where the name of output feature channel
            size is `mid_channel` for the third conv layers. Default: Inflate3D.
        inflate (int): Whether to inflate kernel.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        down_sample (nn.Module | None): DownSample layer. Default: None.
        block_idx (int): the id of the block.
        se_ratio (float | None): The reduction ratio of squeeze and excitation
            unit. If set as None, it means not using SE unit. Default: None.
        use_swish (bool): Whether to use swish as the activation function
            before and after the 3x3x3 conv. Default: True.
        drop_connect_rate (float): dropout rate. If equal to 0.0, perform no dropout.
        bottleneck_factor (float): Bottleneck expansion factor for the 3x3x3 conv.
    """
    expansion: int = 1

    def __init__(self,
                 in_channel,
                 out_channel,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 inflate: int = 2,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 block_idx: int = 0,
                 se_ratio: float = 0.0625,
                 use_swish: bool = True,
                 drop_connect_rate: float = 0.0,
                 bottleneck_factor: float = 2.25,
                 **kwargs):

        super(BlockX3D, self).__init__(in_channel=in_channel,
                                       out_channel=out_channel,
                                       mid_channel=int(out_channel * bottleneck_factor),
                                       conv12=conv12,
                                       norm=norm,
                                       down_sample=down_sample,
                                       inflate=inflate,
                                       **kwargs)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self._drop_connect_rate = drop_connect_rate
        if self.use_swish:
            self.swish = Swish()

        self.se_module = None
        if self.se_ratio > 0.0 and (block_idx + 1) % 2:
            self.se_module = SqueezeExcite3D(self.conv12.mid_channel, self.se_ratio)

        self.conv3 = Unit3D(
            in_channels=self.conv12.mid_channel,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=nn.BatchNorm3d,
            activation=None)
        self.conv3.transform_final_bn = True

    def construct(self, x):
        """Defines the computation performed at every call."""
        identity = x

        out = self.conv12(x)
        if self.se_module is not None:
            out = self.se_module(out)
        if self.use_swish:
            out = self.swish(out)

        out = self.conv3(out)

        if self.training and self._drop_connect_rate > 0.0:
            out = drop_path(out, self._drop_connect_rate)

        if self.down_sample:
            identity = self.down_sample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNetX3D(ResNet3D):
    """
    X3D backbone definition.

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (list): The numbers of block in different layers.
        stage_channels (Tuple[int]): Output channel for every res stage.
        stage_strides (Tuple[Tuple[int]]): Stride size for ResNet3D convolutional layer.
        drop_rates (list): list of the drop rate in different blocks. The basic rate at which blocks
            are dropped, linearly increases from input to output blocks.
        down_sample (Optional[nn.Cell]): Residual block in every resblock, it can transfer the input
            feature into the same channel of output. Default: Unit3D.
        bottleneck_factor (float): Bottleneck expansion factor for the 3x3x3 conv.
        fc_init_std (float): The std to initialize the fc layer(s).

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = ResNetX3D(BlockX3D, [3, 5, 11, 7], (24, 48, 96, 192), ((1, 2, 2),(1, 2, 2),
        >>>             (1, 2, 2),(1, 2, 2)), [0.2, 0.3, 0.4, 0.5], Unit3D)
    """

    def __init__(self,
                 block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int],
                 stage_strides: Tuple[Tuple[int]],
                 drop_rates: Tuple[float],
                 down_sample: Optional[nn.Cell] = Unit3D,
                 bottleneck_factor: float = 2.25
                 ):
        super(ResNetX3D, self).__init__(block=block,
                                        layer_nums=layer_nums,
                                        stage_channels=stage_channels,
                                        stage_strides=stage_strides,
                                        down_sample=down_sample)
        self.in_channels = stage_channels[0]
        self.base_channels = 24
        self.conv1 = nn.SequentialCell([Unit3D(3,
                                               self.base_channels,
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               norm=None,
                                               activation=None),
                                        Unit3D(self.base_channels,
                                               self.base_channels,
                                               kernel_size=(5, 1, 1),
                                               stride=(1, 1, 1))])
        self.layer1 = self._make_layer(
            block,
            stage_channels[0],
            layer_nums[0],
            stride=tuple(stage_strides[0]),
            inflate=2,
            drop_connect_rate=drop_rates[0],
            block_idx=list(range(layer_nums[0])))
        self.layer2 = self._make_layer(
            block,
            stage_channels[1],
            layer_nums[1],
            stride=tuple(stage_strides[1]),
            inflate=2,
            drop_connect_rate=drop_rates[1],
            block_idx=list(range(layer_nums[1])))
        self.layer3 = self._make_layer(
            block,
            stage_channels[2],
            layer_nums[2],
            stride=tuple(stage_strides[2]),
            inflate=2,
            drop_connect_rate=drop_rates[2],
            block_idx=list(range(layer_nums[2])))
        self.layer4 = self._make_layer(
            block,
            stage_channels[3],
            layer_nums[3],
            stride=tuple(stage_strides[3]),
            inflate=2,
            drop_connect_rate=drop_rates[3],
            block_idx=list(range(layer_nums[3])))
        self.conv5 = Unit3D(stage_channels[-1],
                            int(math.ceil(stage_channels[-1] * bottleneck_factor)),
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self._init_weights()

    def _init_weights(self, zero_init_final_bn=True):
        """
        Performs ResNet style weight initialization.

        Args:
            fc_init_std (float): the expected standard deviation for fc layer.
            zero_init_final_bn (bool): if True, zero initialize the final bn for
                every bottleneck.

        Follow the initialization method proposed in:
        {He, Kaiming, et al.
        "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification."
        arXiv preprint arXiv:1502.01852 (2015)}
        """
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv3d):
                m.weight.set_data(initializer(
                    HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
            elif isinstance(m, Unit3D):
                flag = False
                if (hasattr(m, "transform_final_bn")
                        and m.transform_final_bn and zero_init_final_bn):
                    flag = True
                for _, n in m.cells_and_names():
                    if isinstance(n, nn.BatchNorm3d):
                        if flag:
                            batchnorm_weight = 0.0
                        else:
                            batchnorm_weight = 1.0
                        if n.bn2d.gamma is not None:
                            fill = ops.Fill()
                            n.bn2d.gamma.set_data(fill(
                                mindspore.float32, n.bn2d.gamma.shape, batchnorm_weight))
                        if n.bn2d.beta is not None:
                            zeroslike = ops.ZerosLike()
                            n.bn2d.beta.set_data(zeroslike(n.bn2d.beta))

    def construct(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        return x


class X3DHead(nn.Cell):
    """
    x3d head architecture.

    Args:
        input_channel (int): The number of input channel.
        out_channel (int): The number of inner channel. Default: 2048.
        num_classes (int): Number of classes. Default: 400.
        dropout_rate (float): Dropout keeping rate, between [0, 1]. Default: 0.5.

    Returns:
        Tensor

    Examples:
        >>> head = X3DHead(input_channel=432, out_channel=2048, num_classes=400, dropout_rate=0.5)
    """

    def __init__(self,
                 pool_size,
                 input_channel,
                 out_channel=2048,
                 num_classes=400,
                 dropout_rate=0.5,
                 ):
        super(X3DHead, self).__init__()

        self.avg_pool = AvgPool3D(pool_size)

        self.lin_5 = nn.Conv3d(
            input_channel,
            out_channel,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=0,
            has_bias=False
        )

        self.lin_5_relu = nn.ReLU()

        self.dense = DropoutDense(
            input_channel=out_channel,
            out_channel=num_classes,
            has_bias=True,
            keep_prob=dropout_rate)

        self.softmax = nn.Softmax(4)
        self.transpose = ops.Transpose()

    def construct(self, x):

        x = self.avg_pool(x)

        x = self.lin_5(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = self.transpose(x, (0, 2, 3, 4, 1))

        x = self.dense(x)

        if not self.training:
            x = self.softmax(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x


class x3d(nn.Cell):
    """
    x3d architecture.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self,
                 block: Type[BlockX3D],
                 depth_factor: float,
                 num_frames: int,
                 train_crop_size: int,
                 num_classes: int,
                 dropout_rate: float,
                 bottleneck_factor: float = 2.25,
                 eval_with_clips: bool = False):
        super(x3d, self).__init__()

        block_basis = [1, 2, 5, 3]
        stage_channels = (24, 48, 96, 192)
        stage_strides = ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2))
        drop_rates = (0.2, 0.3, 0.4, 0.5)
        layer_nums = []
        for item in block_basis:
            nums = int(math.ceil(item * depth_factor))
            layer_nums.append(nums)
        spat_sz = int(math.ceil(train_crop_size / 32.0))
        pool_size = [num_frames, spat_sz, spat_sz]
        input_channel = int(math.ceil(192 * bottleneck_factor))

        self.num_frames = num_frames
        self.eval_with_clips = eval_with_clips
        self.softmax = nn.Softmax()
        self.transpose = ops.Transpose()

        self.backbone = ResNetX3D(block=block, layer_nums=layer_nums, stage_channels=stage_channels,
                                  stage_strides=stage_strides, drop_rates=drop_rates)
        self.head = X3DHead(pool_size=pool_size, input_channel=input_channel, out_channel=2048,
                            num_classes=num_classes, dropout_rate=dropout_rate)

    def construct(self, x):

        if not self.eval_with_clips:
            x = self.backbone(x)
            x = self.head(x)
        else:
            # use for 10-clip eval
            b, c, n, h, w = x.shape        
            if n > self.num_frames:
                x = x.reshape(b, c, -1, self.num_frames, h, w)
                x = self.transpose(x, (2, 0, 1, 3, 4, 5))
                x = x.reshape(-1, c, self.num_frames, h, w)        

            x = self.backbone(x)
            x = self.head(x)

            if n > self.num_frames:
                x = self.softmax(x)
                x = x.reshape(-1, b, 400)
                x = x.mean(axis=0, keep_dims=False)

        return x


@ClassFactory.register(ModuleType.MODEL)
def x3d_m(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 2.2,
          num_frames: int = 16,
          train_crop_size: int = 224,
          eval_with_clips: bool = False,
          ):
    """
    X3D middle model.

    Christoph Feichtenhofer. "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730

    Args:
        num_classes (int): the channel dimensions of the output.
        dropout_rate (float): dropout rate. If equal to 0.0, perform no
            dropout.
        depth_factor (float): Depth expansion factor.
        num_frames (int): The number of frames of the input clip.
        train_crop_size (int): The spatial crop size for training.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindvision.msvideo.models import x3d_m
        >>>
        >>> network = x3d_m()
        >>> input_x = Tensor(np.random.randn(1, 3, 16, 224, 224).astype(np.float32))
        >>> out = network(input_x)
        >>> print(out.shape)
        (1, 400)

    About x3d: Expanding Architectures for Efficient Video Recognition.

    .. code-block::

        @inproceedings{x3d2020,
            Author    = {Christoph Feichtenhofer},
            Title     = {{X3D}: Progressive Network Expansion for Efficient Video Recognition},
            Booktitle = {{CVPR}},
            Year      = {2020}
        }
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size,
               num_classes, dropout_rate, eval_with_clips=eval_with_clips)


@ClassFactory.register(ModuleType.MODEL)
def x3d_s(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 2.2,
          num_frames: int = 13,
          train_crop_size: int = 160,
          eval_with_clips: bool = False,
          ):
    """
    X3D small model.
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size,
               num_classes, dropout_rate, eval_with_clips=eval_with_clips)


@ClassFactory.register(ModuleType.MODEL)
def x3d_xs(num_classes: int = 400,
           dropout_rate: float = 0.5,
           depth_factor: float = 2.2,
           num_frames: int = 4,
           train_crop_size: int = 160,
           eval_with_clips: bool = False,
           ):
    """
    X3D x-small model.
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size,
               num_classes, dropout_rate, eval_with_clips=eval_with_clips)


@ClassFactory.register(ModuleType.MODEL)
def x3d_l(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 5.0,
          num_frames: int = 16,
          train_crop_size: int = 312,
          eval_with_clips: bool = False,
          ):
    """
    X3D large model.
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size,
               num_classes, dropout_rate, eval_with_clips=eval_with_clips)
