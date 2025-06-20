from typing import Callable, Optional, Type, Union, List, Any, Tuple

import torch
import torch.nn as nn

from .activation import get_activation_class
from .utils import program_conv_filters


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock3D(nn.Module):
    expansion: int =1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int =1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_channels != 64:
            raise ValueError("BasicBlock3D only supports groups=1 and base_channels=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock3D")

        self.conv1 = conv3x3x3(
            in_planes=in_channels,
            out_planes=out_channels,
            stride=stride
        )
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation()

        self.conv2 = conv3x3x3(
            in_planes=out_channels,
            out_planes=out_channels,
            stride=stride
        )
        self.norm2 = norm_layer(out_channels)
        self.act2 = activation()
        self.downsample = downsample
        self.stride=stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act2(out)

        return out

class BottleneckBlock3D(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_channels: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(out_channels * (base_channels / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(
            in_planes=in_channels,
            out_planes=out_channels,
        )
        self.norm1 = norm_layer(width)
        self.act1 = activation()

        self.conv2 = conv3x3x3(
            in_planes=out_channels,
            out_planes=out_channels,
            stride=stride
        )
        self.norm2 = norm_layer(width)
        self.act2 = activation()

        self.conv3 = conv3x3x3(
            in_planes=in_channels,
            out_planes=out_channels * self.expansion,

        )
        self.norm3 = norm_layer(out_channels * self.expansion)
        self.act3 = activation()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act3(out)

        return out

class ResNet3D(nn.Module):
    def __init__(
            self,
            block: str,
            conv_layers: List[int],
            in_channels: int,
            out_dims: int,
            seq_size: int,
            base_channels: int,
            fc_stages: int,
            dropout: float = 0.1,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: str = "relu",
            base_pool: str = "max",
            final_pool: str = "average",
            **kwargs,
    ) -> None:
        super().__init__()

        if block not in ["basic", "bottleneck"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(block) " f"receives one of ['basic', 'bottleneck'].")

        if final_pool not in ["average", "max"] or base_pool not in ["average", "max"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(final_pool, base_pool) both "
                f"receives one of ['average', 'max']."
            )

        if fc_stages < 1:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(fc_stages) receives " f"an integer equal to ore more than 1."
            )

        self.fc_stages = fc_stages

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.zero_init_residual = zero_init_residual

        self.current_channels = base_channels
        self.groups = groups
        self.base_channels = width_per_group
        self.groups = groups

        if base_pool == "average":
            self.base_pool = nn.AvgPool3d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool3d

        if block == "basic":
            block = BasicBlock3D
            conv_filter_list = [
                {"kernel_size": 7},
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
            ]
        else:  # bottleneck case
            block = BottleneckBlock3D
            conv_filter_list = [
                {"kernel_size": 7},
                {"kernel_size": 3},
                {"kernel_size": 3},
                {"kernel_size": 3},
                {"kernel_size": 3},
            ]

        self.sequence_length = seq_size
        self.output_length = program_conv_filters(
            sequence_length=seq_size,
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            class_name=self.__class__.__name__,
        )

        cf = conv_filter_list[0]
        input_stage = []
        if cf["pool"] > 1:
            input_stage.append(self.base_pool(cf["pool"]))

        input_stage.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.current_channels,
                kernel_size=(cf["kernel_size"],cf["kernel_size"],cf["kernel_size"]),
                stride=(cf["stride"],cf["stride"],cf["stride"]),
                padding=cf["kernel_size"] // 2,
                bias=False,
            )
        )
        input_stage.append(norm_layer(self.current_channels))
        input_stage.append(self.nn_act())
        self.input_stage = nn.Sequential(*input_stage)

        cf = conv_filter_list[1]
        self.conv_stage1 = self._make_conv_stage(
            block,
            channels=base_channels,
            blocks=conv_layers[0],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )
        cf = conv_filter_list[2]
        self.conv_stage2 = self._make_conv_stage(
            block,
            channels=2 * base_channels,
            blocks=conv_layers[1],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )
        cf = conv_filter_list[3]
        self.conv_stage3 = self._make_conv_stage(
            block,
            channels=4 * base_channels,
            blocks=conv_layers[2],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )
        cf = conv_filter_list[4]
        self.conv_stage4 = self._make_conv_stage(
            block,
            channels=8 * base_channels,
            blocks=conv_layers[3],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )

        if final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool3d(1)
        elif final_pool == "max":
            self.final_pool = nn.AdaptiveMaxPool3d(1)

        fc_stage = []

        for i in range(fc_stages - 1):
            layer = nn.Sequential(
                nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                nn.Dropout(p=dropout),
                nn.BatchNorm1d(self.current_channels // 2),
                self.nn_act(),
            )
            self.current_channels = self.current_channels // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(self.current_channels, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.xavier_normal_(m.weight)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock3D):
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_conv_stage(
        self,
        block: Type[Union[BasicBlock3D, BottleneckBlock3D]],
        channels: int,
        blocks: int,
        stride: int = 1,
        pre_pool: int = 1,
        activation=nn.ReLU,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.current_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.current_channels,
                    out_channels=channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(channels * block.expansion),
            )

        conv_layers = []

        if pre_pool > 1:
            conv_layers.append(self.base_pool(pre_pool))

        conv_layers.append(
            block(
                in_channels=self.current_channels,
                out_channels=channels,
                base_channels=self.base_channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                activation=activation,
            )
        )

        self.current_channels = channels * block.expansion
        for _ in range(1, blocks):
            conv_layers.append(
                block(
                    in_channels=self.current_channels,
                    out_channels=channels,
                    base_channels=self.base_channels,
                    groups=self.groups,
                    stride=1,
                    activation=activation,
                )
            )

        return nn.Sequential(*conv_layers)

    def get_output_length(self):
        return self.output_length

    def get_dims_from_last(self, target_from_last: int):
        l = self.fc_stages - target_from_last
        return self.fc_stage[l][0].in_features

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, target_from_last: int = 0):
        N, _, D, H, W = x.size()

        x = self.input_stage(x)

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)

        x = self.final_pool(x)
        x = x.reshape((N, -1))

        if target_from_last == 0:
            x = self.fc_stage(x)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(
                    f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                    f"an integer equal to or smaller than fc_stages={self.fc_stages}."
                )

            for l in range(self.fc_stages - target_from_last):
                x = self.fc_stage[l](x)
        return x

    def forward(self, x):
        x = self.compute_feature_embedding(x)
        return x