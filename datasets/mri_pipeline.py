import time
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_variable_repr(a: dict):
    return f"({', '.join([f'{k}={v!r}' for k, v in a.items() if not (k.startswith('_') or k == 'training')])})"

#####################
#   MRI PREPROCESS  #
#####################
class MriSpatialPad(nn.Module):
    """
    Apply spatial padding to a 3D volume tensor to reach a target spatial size.
    Supports 'symmetric' or 'end' padding mode.
    """

    def __init__(self, spatial_size, mode='constant', method='symmetric', value=0):
        super().__init__()
        self.spatial_size = spatial_size  # e.g., [128, 128, 128]
        self.mode = mode  # 'constant', 'reflect', etc. (from torch.nn.functional.pad)
        self.method = method  # 'symmetric' or 'end'
        self.value = value  # padding value for 'constant' mode

    def forward(self, sample):
        volume = sample['volume']
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)

        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        # Get original shape
        D, H, W = volume.shape
        target_D, target_H, target_W = self.spatial_size

        pad_d = max(target_D - D, 0)
        pad_h = max(target_H - H, 0)
        pad_w = max(target_W - W, 0)

        if self.method == 'symmetric':
            pad = [
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2,
            ]
        elif self.method == 'end':
            pad = [
                0, pad_w,
                0, pad_h,
                0, pad_d
            ]
        else:
            raise ValueError(f"Unknown padding method: {self.method}")

        # F.pad expects NCDHW or NDH
        padded = torch.nn.functional.pad(volume, pad, mode=self.mode, value=self.value)
        sample['volume'] = padded
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"

class MriResize(nn.Module):
    """
    Resize a 3D MRI volume to the target spatial size using interpolation.
    Equivalent to monai.transforms.Resize but in torch.nn.Module format.
    """

    def __init__(self, spatial_size, mode='trilinear', align_corners=False):
        """
        Args:
            spatial_size (tuple or list): The desired output size (D, H, W).
            mode (str): Interpolation mode. Default: 'trilinear'.
            align_corners (bool): Align corners when interpolating. Default: False.
        """
        super().__init__()
        self.spatial_size = (spatial_size,spatial_size,spatial_size)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, sample):
        volume = sample['volume']

        # Convert to tensor if needed
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)

        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        # Add fake batch and channel dimensions: [1, 1, D, H, W]
        volume = volume.unsqueeze(0).unsqueeze(0).float()

        # Resize
        resized = F.interpolate(volume, size=self.spatial_size, mode=self.mode, align_corners=self.align_corners)

        # Remove fake dimensions
        sample['volume'] = resized.squeeze(0).squeeze(0)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


#Todo : Normalize during it's tensor
class MriNormalize(torch.nn.Module):
    """Z-score Normalize MRI image by its internal statistics."""
    def __init__(self, eps=1e-8, mri_norm_type='z_score'):
        super().__init__()
        self.eps = eps
        self.norm_type = mri_norm_type

    def forward(self,sample):
        volume = sample['volume']
        mask = volume > 0
        norm = np.zeros_like(volume)
        if self.norm_type == 'z_score':
            mean = np.mean(volume[mask])
            std = np.std(volume[mask])
            norm[mask] = (volume[mask] - mean) / std
        elif self.norm_type == 'min_max':
            min_val = np.min(volume[mask])
            max_val = np.max(volume[mask])
            norm[mask] = (volume[mask] - min_val) / (max_val - min_val)
        sample['volume'] = norm
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"


#####################
#   MRI TRANSFORM   #
#####################

class MriAdditiveGaussianNoise(torch.nn.Module):
    """Additive White Gaussian Noise to MRI volume"""
    def __init__(self, mean=0.0, std=1e-2):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        volume = sample['volume']
        noise = torch.normal(
            mean=torch.ones_like(volume) * self.mean,
            std=torch.ones_like(volume) * self.std,
        )
        sample['volume'] = volume + noise
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"

class MriToTensor(object):
    """Convert MRI numpy array in sample to Tensors."""

    @staticmethod
    def _volume_to_tensor(volume):
        if isinstance(volume, (np.core.memmap,)):
            return torch.tensor(volume).to(dtype=torch.float32)
        return torch.from_numpy(volume).to(dtype=torch.float32)

    def __call__(self, sample):
        volume = sample["volume"]

        if isinstance(volume, (np.ndarray,)):
            sample["volume"] = self._volume_to_tensor(volume)
        elif isinstance(volume, (list,)):
            volumes = []
            for v in volume:
                volumes.append(self._volume_to_tensor(v))
            sample["volume"] = volumes
        elif isinstance(volume, (torch.Tensor,)):
            sample["volume"] = volume
        else:
            raise ValueError(
                f'{self.__class__.__name__}.__call__(sample["volume"]) needs to be set to np.ndarray ' f"or their list"
            )

        # sample["age"] = torch.tensor(sample["age"], dtype=torch.float32)
        # if "class_label" in sample.keys():
        #     sample["class_label"] = torch.tensor(sample["class_label"])

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"

class MriToDevice(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, sample):
        sample['volume'] = sample['volume'].to(self.device)
        if 'class_label' in sample.keys():
            sample['class_label'] = sample['class_label'].to(self.device)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{make_variable_repr(self.__dict__)}"

def emm_collate_fn(batch):
    batched_sample = {k: [] for k in batch[0].keys()}

    for sample in batch:
        if isinstance(sample["signal"], (np.ndarray,)) or torch.is_tensor(sample["signal"]):
            for k in sample.keys():
                batched_sample[k] += [sample[k]]

        elif isinstance(sample["signal"], (list,)):
            multiple = len(sample["signal"])

            for s in sample["signal"]:
                batched_sample["signal"] += [s]

            for k in sample.keys():
                if k not in ["signal", "crop_timing"]:
                    batched_sample[k] += multiple * [sample[k]]
                elif k == "crop_timing":
                    batched_sample[k] += [*sample[k]]

    batched_sample["signal"] = torch.stack(batched_sample["signal"])
    batched_sample["volume"] = torch.stack(batched_sample["volume"])
    batched_sample["age"] = torch.stack(batched_sample["age"])
    if "class_label" in batched_sample.keys():
        batched_sample["class_label"] = torch.stack(batched_sample["class_label"])

    return batched_sample
