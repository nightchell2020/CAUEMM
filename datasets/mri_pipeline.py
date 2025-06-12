import time
from packaging import version

import numpy as np
import torch
import torchaudio


def make_variable_repr(a: dict):
    return f"({', '.join([f'{k}={v!r}' for k, v in a.items() if not (k.startswith('_') or k == 'training')])})"

class MriDropChannels(object):
    def __init__(self, index):
        self.drop_index = index

    def __call__(self, sample):
        image = sample["image"]

        return sample

class MriResizing(object):
    def __call__(self, sample):
        return sample

class MriNormalize(torch.nn.Module):
    """Z-score Normalize MRI image by its internal statistics."""
    def __init__(self, eps=1e-8, norm_type='z_score'):
        super().__init__()
        self.eps = eps
        self.norm_type = norm_type

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

class MriToTensor(object):
    """Convert MRI numpy array in sample to Tensors."""

    @staticmethod
    def _volume_to_tensor(volume):
        if isinstance(volume, (np.core.memmap,)):
            return torch.tensor(volume).to(dtype=torch.float32)
        return torch.from_numpy(volume).to(dtype=torch.float32)

    def __call__(self, sample):
        signal = sample["volume"]

        if isinstance(signal, (np.ndarray,)):
            sample["volume"] = self._volume_to_tensor(signal)
        elif isinstance(signal, (list,)):
            volumes = []
            for s in signal:
                volumes.append(self._volume_to_tensor(s))
            sample["volume"] = volumes
        else:
            raise ValueError(
                f'{self.__class__.__name__}.__call__(sample["volume"]) needs to be set to np.ndarray ' f"or their list"
            )

        sample["age"] = torch.tensor(sample["age"], dtype=torch.float32)
        if "class_label" in sample.keys():
            sample["class_label"] = torch.tensor(sample["class_label"])

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
