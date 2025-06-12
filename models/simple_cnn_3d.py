import numpy as np
import torch
import torch.nn as nn

from .activation import get_activation_class
from .activation import get_activation_functional


class Ieracitano3DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        fc_stages: int,
        seq_length: int,
        use_age: str,
        base_channels: int = 16,
        base_pool: str = "max",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        if use_age != "no":
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) accepts for only 'no'.")

        if fc_stages != 1:
            raise ValueError(f"{self.__class__.__name__}.__init__(fc_stages) accepts for only 1.")

        self.sequence_length = seq_length
        self.fc_stages = fc_stages

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
            kernel_size=(3, 3, 3),
            padding=1,
            stride=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.pool1 = self.base_pool(kernel_size=(2, 2, 2))

        # 예시: 입력 크기 (N, C=1, D=10, H=32, W=32) → conv → pool → flatten
        # 실질적으로 D,H,W 사이즈에 맞게 계산해야 함
        flattened_dim = base_channels * (32 * 32 * 32)  # 예시 기준: D/2=5, H/2=16, W/2=16
        fc_stage = []
        fc_stage.append(nn.Linear(flattened_dim, 300))
        fc_stage.append(nn.BatchNorm1d(300))
        fc_stage.append(self.nn_act())

        fc_stage.append(nn.Linear(300, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

        self.output_length = 300  # 첫 FC 차원 출력
        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def get_dims_from_last(self, target_from_last: int):
        l = self.fc_stages - target_from_last
        return self.fc_stage[l * 3].in_features

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        N = x.size(0)

        # conv-bn-act-pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.F_act(x)
        x = self.pool1(x)

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

    def forward(self, x, age):
        x = self.compute_feature_embedding(x, age)
        return x