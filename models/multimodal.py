import numpy as np
import torch
import torch.nn as nn

from .activation import get_activation_class
from .activation import get_activation_functional


class EMMNet(nn.Module):
    def __init__(
            self,
            mri_model,
            eeg_model,
            fc_stages: int,
            fusion='concat',
            **kwargs,
        ):
        super().__init__()
        self.mri_model = mri_model
        self.eeg_model = eeg_model
        self.fc_stages = fc_stages
        self.fusion = fusion

        fc_stage = []
        for i in range(fc_stages - 1):
            layer = nn.Sequential(
                nn.Linear(1000, 300, bias=False),
                nn.Dropout(),
                nn.BatchNorm1d(300),
                self.nn_act()
            )
            fc_stage.append(layer)

        self.fc_stage = nn.Sequential(*fc_stage)
        # Todo: programming output_length?
        self.output_length = 10

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def get_num_fc_stages(self):
        return self.fc_stages

    def get_output_length(self):
        return self.output_length

    def compute_feature_embedding(self, x3d, x1d, age, target_from_last: int = 0):
        feat3d = self.cnn3d(x3d)
        feat1d = self.resnet1d(x1d, age)

        if self.fusion == "concat":
            fused = torch.cat([feat3d, feat1d], dim=1)
        elif self.fusion == "add":
            fused = feat3d + feat1d
        else:
            raise ValueError("Invalid fusion method")

        if target_from_last == 0:
            x = self.fc_stage(fused)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(
                    f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                    f"an integer equal to or smaller than fc_stages={self.fc_stages}."
                )

            for l in range(self.fc_stages - target_from_last):
                x = self.fc_stage[l](fused)

        return x

    def forward(self, x3d, x1d, age):
        x = self.compute_feature_embedding(x3d, x1d, age)
        return x

