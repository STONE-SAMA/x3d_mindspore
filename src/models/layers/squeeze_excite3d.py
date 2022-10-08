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
"""Custom operators."""

from typing import Optional, Union

from mindspore import nn

from src.utils.class_factory import ClassFactory, ModuleType
from src.models.layers.adaptiveavgpool3d import AdaptiveAvgPool3D
from src.models.layers.swish import Swish

__all__ = ['SqueezeExcite3D']


def make_divisible(v: float,
                   divisor: int,
                   min_value: Optional[int] = None
                   ):
    """
    It ensures that all layers have a channel number that is divisible by 8.

    Args:
        v (int): original channel of kernel.
        divisor (int): Divisor of the original channel.
        min_value (int, optional): Minimum number of channels.

    Returns:
        Number of channel.

    Examples:
        >>> _make_divisible(32, 8)
    """

    if not min_value:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@ClassFactory.register(ModuleType.LAYER)
class SqueezeExcite3D(nn.Cell):
    """
    Squeeze-and-Excitation (SE) block implementation.

    Args:
        dim_in (int): the channel dimensions of the input.
        ratio (float): the channel reduction ratio for squeeze.
        act_fn (Union[str, nn.Cell]): the activation of conv_expand: Default: Swish.

    Returns:
        Tensor
    """

    def __init__(self, dim_in, ratio, act_fn: Union[str, nn.Cell] = Swish):
        super(SqueezeExcite3D, self).__init__()
        self.avg_pool = AdaptiveAvgPool3D((1, 1, 1))
        v = dim_in * ratio
        dim_fc = make_divisible(v=v, divisor=8)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, has_bias=True)
        self.fc1_act = nn.ReLU() if act_fn else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, has_bias=True)
        self.fc2_sig = nn.Sigmoid()

    def construct(self, x):
        x_in = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_sig(x)
        return x_in * x
