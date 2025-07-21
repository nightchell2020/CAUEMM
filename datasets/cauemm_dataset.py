import os
import json
from copy import deepcopy
import pyedflib

import numpy as np
import pyarrow.feather as feather
import torch
import nibabel as nib
from torch.utils.data import Dataset

class CauEegMriMultiModalDataset(Dataset):
    """PyTorch Dataset Class for CAUEEGMRIMultiModal Dataset.

    Args:


    """

    def __init__(
        self,
        root_dir: str,
        data_list: list,
        load_event: bool,
        eeg_file_format: str = "memmap",
        use_prefix_signal: bool = True,
        transform = None,
    ):
        if eeg_file_format not in ["edf", "feather", "memmap", "np"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"eeg files must be set to one of 'edf', 'feather', 'memmap' and 'np'"
            )
        self.root_dir = root_dir
        self.data_list = data_list
        self.load_event = load_event
        self.eeg_file_format = eeg_file_format
        self.mri_file_format = 'nii'
        self.use_prefix_signal = use_prefix_signal
        self.eeg_transform = transform[0]
        self.mri_transform = transform[1]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # annotation
        sample = deepcopy(self.data_list[idx])

        # eeg_signal
        sample["signal"] = self._read_signal(sample)

        # mri_volume
        sample["volume"] = self._read_volume(sample)

        # event
        if self.load_event:
            sample["event"] = self._read_event(sample)

        if self.eeg_transform:
            sample = self.eeg_transform(sample)
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

    def _read_signal(self, anno):
        if self.eeg_file_format == "edf":
            return self._read_edf(anno)
        elif self.eeg_file_format == "feather":
            return self._read_feather(anno)
        else:
            return self._read_memmap(anno)

    def _read_edf(self, anno):
        prefix = "CAUEEG/caueeg-dataset/signal/edf/" if self.use_prefix_signal else ""
        edf_file = os.path.join(self.root_dir, prefix, f"{anno['serial']}.edf")
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        return signal

    def _read_feather(self, anno):
        prefix = "CAUEEG/caueeg-dataset/signal/feather/" if self.use_prefix_signal else ""
        fname = os.path.join(self.root_dir, prefix, f"{anno['serial']}.feather")
        df = feather.read_feather(fname)
        return df.values.T

    def _read_memmap(self, anno):
        prefix = "CAUEEG/caueeg-dataset/signal/memmap/" if self.use_prefix_signal else ""
        fname = os.path.join(self.root_dir, prefix, f"{anno['serial']}.dat")
        signal = np.memmap(fname, dtype="int32", mode="r").reshape(21, -1)
        return signal

    def _read_np(self, anno):
        prefix = "CAUEEG/caueeg-dataset/signal/np/" if self.use_prefix_signal else ""
        fname = os.path.join(self.root_dir, prefix, f"{anno['serial']}.npy")
        return np.load(fname)

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

