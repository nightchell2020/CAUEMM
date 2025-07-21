import numpy as np
import torch
import torch.nn as nn

from .activation import get_activation_class
from .activation import get_activation_functional

class Simple3DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        fc_stages: int,
        image_size: int,
        base_channels: int = 16,
        base_pool: str = "max",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        if fc_stages != 1:
            raise ValueError(f"{self.__class__.__name__}.__init__(fc_stages) accepts for only 1.")

        self.image_size = image_size
        self.fc_stages = fc_stages
        self.out_dims = out_dims
        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.F_act = get_activation_functional(activation, class_name=self.__class__.__name__)

        if base_pool == "average":
            self.base_pool = nn.AvgPool3d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool3d
        else:
            raise ValueError(f"{self.__class__.__name__}.__init__: Invalid base_pool '{base_pool}'")

        self.conv1 = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=(7, 7, 7),
            padding=1,
            stride=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.pool1 = self.base_pool(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(
            base_channels,
            base_channels*2,
            kernel_size=(5, 5, 5),
            padding=1,
            stride=1,
            bias=True,
        )
        self.bn2 = nn.BatchNorm3d(base_channels*2)
        self.pool2 = self.base_pool(kernel_size=(2, 2, 2))
        # input data = (Batch, Channel=1, D, W, H) where D,H,W = self.image_size

        flattened_dim = base_channels*2 * (14 * 14 * 14)
        fc_stage = []
        fc_stage.append(nn.Linear(flattened_dim, 300))
        fc_stage.append(nn.BatchNorm1d(300))
        fc_stage.append(self.nn_act())

        fc_stage.append(nn.Linear(300, self.out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)


        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def get_output_length(self):
        return self.out_dims

    def get_dims_from_last(self, target_from_last: int):
        l = self.fc_stages - target_from_last
        return self.fc_stage[l * 3].in_features

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, target_from_last: int = 0):
        # x : (N, C_in=1, D, H, W)
        N = x.size(0)

        # conv-bn-act-pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.F_act(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.F_act(x)
        x = self.pool2(x)

        x = x.view(N, -1)

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