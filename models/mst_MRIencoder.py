import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from torch.utils.checkpoint import checkpoint
from typing import Callable, Optional, Type, Union, List, Any, Tuple
from .activation import get_activation_class


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x  # Residual
        return out


# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)

    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out


# Image Encoder module
class ImageEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            base_channels: int,
            out_dims: int,
            fc_stages: int,
            dropout: float = 0.1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: str = "relu",
            base_pool: str = "max",
            final_pool: str = "average",
    ):
        super(ImageEncoder, self).__init__()

        if final_pool not in ["average", "max"] or base_pool not in ["average", "max"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(final_pool, base_pool) both "
                f"receives one of ['average', 'max']."
            )

        if fc_stages < 1:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(fc_stages) receives " f"an integer equal to ore more than 1."
            )

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.current_channels = base_channels
        self.fc_stages = fc_stages
        self.output_length = out_dims

        input_stage = []
        input_stage.append(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )
        input_stage.append(norm_layer(self.current_channels))
        input_stage.append(self.nn_act())
        self.input_stage = nn.Sequential(*input_stage)

        self.RDB1_c1 = RDB(self.current_channels, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB2_c1 = RDB(self.current_channels, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB3_c1 = RDB(self.current_channels, nDenselayer=4, growthRate=32, norm='BN')

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


    def get_output_length(self):
        return self.output_length

    def get_dims_from_last(self, target_from_last: int):
        l = self.fc_stages - target_from_last
        return self.fc_stage[l][0].in_features

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, target_from_last: int = 0):
        N, _ , D, H, W = x.size()

        x = self.input_stage(x)

        x = checkpoint(self.RDB1_c1,x)
        x = F.avg_pool3d(x, 2)
        x = checkpoint(self.RDB2_c1,x)
        x = F.avg_pool3d(x, 2)
        x = checkpoint(self.RDB3_c1,x)
        x = F.avg_pool3d(x, 2)

        x = self.final_pool(x)
        x = x.reshape((N,-1))

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

    def forward(self,x):
        x = self.compute_feature_embedding(x)
        return x