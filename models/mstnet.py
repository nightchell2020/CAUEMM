"""
Modified from:
    - MSTNet
    - https://github.com/JustlfC03/MSTNet/tree/main
    - VGG paper: https://arxiv.org/abs/1409.1556
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Type, Union, List, Any, Tuple
from .activation import get_activation_class
from torch.utils.checkpoint import checkpoint


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

class ImageEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_dim: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            base_pool: str = "max",
            activation: str = "relu",
            **kwargs,
    ) -> None:
        super().__init__()

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        if base_pool == "average":
            self.base_pool = nn.AvgPool3d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool3d

        input_stage = []
        input_stage.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )
        input_stage.append(norm_layer(self.current_channels))
        input_stage.append(self.nn_act())
        self.input_stage = nn.Sequential(*input_stage)

        self.conv_stage1 = RDB(hidden_dim, nDenselayer=4, growthRate=32, norm='BN')
        self.conv_stage2 = RDB(hidden_dim, nDenselayer=4, growthRate=32, norm='BN')
        self.conv_stage3 = RDB(hidden_dim, nDenselayer=4, growthRate=32, norm='BN')

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

        x = checkpoint(self.conv_stage1,x,use_reentrant=False)
        x = F.avg_pool3d(x,2)
        x = checkpoint(self.conv_stage2,x,use_reentrant=False)
        x = F.avg_pool3d(x, 2)
        x = checkpoint(self.conv_stage3,x,use_reentrant=False)
        x = F.avg_pool3d(x, 2)

        return x

    def forward(self, x):
        x = self.compute_feature_embedding(x)
        return x


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len=100, pred_len=100, top_k=5, d_ff=32, d_model=32, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, C, L = x.size()      # Batch, Channel, Length
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], x.shape[1], (length - (self.seq_len + self.pred_len))]).to(x.device)
                out = torch.cat([x, padding], dim=-1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, C, length // period, period).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.reshape(B, C, -1)
            res.append(out[:, :, :(self.seq_len + self.pred_len)])
        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, C, L, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res

class EEGEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_dims: int,
            seq_length: int,
            base_channels: int,
            use_age: str,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: str = "relu",
            base_pool: str = "max",
            final_pool: str = "average",
            **kwargs,
    ):
        super().__init__()
        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) " f"receives one of ['fc', 'conv', 'no'].")

        if self.use_age == "conv":
            in_channels += 1

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer


        self.sequence_length = seq_length
        # input_stage = []
        # input_stage.append(
        #     nn.Conv1d(
        #         in_channels=in_channels,
        #         out_channels=base_channels,
        #         kernel_size=41,
        #         stride=1,
        #         padding=20,
        #         bias=False,
        #     )
        # )
        # input_stage.append(norm_layer(self.current_channels))
        # input_stage.append(self.nn_act())
        # self.input_stage = nn.Sequential(*input_stage)

        self.TimesNet = nn.ModuleList([TimesBlock() for _ in range(3)])

        self.conv_stage1 = self._make_conv_stage(
            in_channels=base_channels,
            out_channels=2*base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_stage2 = self._make_conv_stage(
            in_channels=2*base_channels,
            out_channels=4*base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_stage3 = self._make_conv_stage(
            in_channels=4*base_channels,
            out_channels=2*base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_stage4 = self._make_conv_stage(
            in_channels=2*base_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.fc_stage = nn.Linear(base_channels, out_dims)
        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.xavier_normal_(m.weight)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _make_conv_stage(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride,
            padding,
    ) -> nn.Sequential:

        conv_layers = []
        conv_layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        conv_layers.append(
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        return nn.Sequential(*conv_layers)

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        N, _, L = x.size()
        if self.use_age == "conv":
            age = age.reshape((N, 1, 1)).expand(N, 1, L)
            x = torch.cat((x, age), dim=1)

        # x = self.input_stage(x)
        x = self.TimesNet(x)

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)

        x = self.fc_stage(x)

        return x

    def forward(self, x, age):
        x = self.compute_feature_embedding(x, age)

        return x