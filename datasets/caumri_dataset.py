import os
import json
from copy import deepcopy
import pyedflib

import numpy as np
import pyarrow.feather as feather
import torch
import nibabel as nib
from torch.utils.data import Dataset

class CauMriDataset(Dataset):
    """PyTorch Dataset Class for CAUEEGMRIMultiModal Dataset.

    Args:


    """

    def __init__(
        self,
        root_dir: str,
        data_list: list,
        load_event: bool,
        use_prefix_signal: bool = True,
        transform = None,
    ):

        self.root_dir = root_dir
        self.data_list = data_list
        self.load_event = load_event
        self.mri_file_format = 'nii'
        self.use_prefix_signal = use_prefix_signal
        self.mri_transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # annotation
        sample = deepcopy(self.data_list[idx])

        # mri_volume
        sample["volume"] = self._read_volume(sample)

        # event
        if self.load_event:
            sample["event"] = self._read_event(sample)

        if self.mri_transform:
            sample = self.mri_transform(sample)
        return sample

    def _read_volume(self, anno):
        if self.mri_file_format == "nii":
            return self._read_nii(anno)
        else:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"support only 'nii' format for now"
            )

    def _read_nii(self,anno):
        prefix = "CAUMRI/caumri-dataset/image/nii/"
        nii_file = os.path.join(self.root_dir, prefix, f"{anno['serial']}.nii")
        nii = nib.load(nii_file)
        image, header = nii.get_fdata(), nii.header
        return image

    def _read_event(self, m):
        fname = os.path.join(self.root_dir, "event", m["serial"] + ".json")
        with open(fname, "r") as json_file:
            event = json.load(json_file)
        return event

