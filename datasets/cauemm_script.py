import json
import math
import os
import pprint

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .cauemm_dataset import CauEegMriMultiModalDataset
from .pipeline import EegAddGaussianNoiseAge, MriSpatialPad, MriResize, MriNormalize, MriToTensor, MriToDevice, \
    emm_collate_fn, MriCenterCrop, MriDropInvalidRange
from .pipeline import EegAdditiveGaussianNoise, EegMultiplicativeGaussianNoise
from .pipeline import EegChannelDropOut
from .pipeline import EegDropChannels, EegChannelDifference
from .pipeline import EegNormalizeAge
from .pipeline import EegNormalizeMeanStd, EegNormalizePerSignal
from .pipeline import EegRandomCrop
from .pipeline import EegResample
from .pipeline import EegSpectrogram
from .pipeline import EegToTensor, EegToDevice


def load_cauemm_config(dataset_path: str):
    """Load the configuration of the CAUMRI dataset.

    Args:
        dataset_path (str): The file path where the dataset files are located.
    """
    try:
        with open(
                os.path.join(dataset_path, "annotation_EMM.json"),
                "r",
        ) as json_file:
            annotation = json.load(json_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caumri_config(dataset_path) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise

    config = {k: v for k, v in annotation.items() if k != "data"}
    return config


def load_cauemm_full_dataset(
        dataset_path: str,
        load_event: bool = True,
        eeg_file_format: str = "memmap",
        transform: list = None,
):
    """Load the whole CAUMRI dataset as a PyTorch dataset instance without considering the target task.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        eeg_file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Preprocessing process to apply during loading signals.

    Returns:
        The PyTorch dataset instance for the entire CAUMRI dataset.
    """
    try:
        with open(
                os.path.join(dataset_path, "annotation_EMM.json"),
                "r",
        ) as json_file:
            annotation = json.load(json_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caumri_full(dataset_path) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise

    emm_dataset = CauEegMriMultiModalDataset(
        dataset_path,
        annotation["data"],
        load_event=load_event,
        eeg_file_format=eeg_file_format,
        transform=transform,
    )

    config = {k: v for k, v in annotation.items() if k != "data"}

    return config, emm_dataset

def load_cauemm_task_datasets(
        dataset_path: str,
        task: str,
        load_event: bool = True,
        eeg_file_format: str = "memmap",
        transform=None,
        verbose=False,
):
    """Load the CAUMRI datasets for the target benchmark task as PyTorch dataset instances.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        task (str): The target task to load among 'dementia' or 'abnormal'.
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        eeg_file_format (str): Determines which file format will be used (default: 'memmap').
        transform (callable): Preprocessing process to apply during loading signals.
        verbose (bool): Whether to print the progress during loading the datasets.

    Returns:
        The PyTorch dataset instances for the train, validation, and test sets for the task and their configurations.
    """
    task = task.lower()
    if task not in [
        "abnormal",
        "dementia",
        "abnormal-no-overlap",
        "dementia-no-overlap",
    ]:
        raise ValueError(
            f"load_cauemm_task_datasets(task) receives the invalid task name: {task}. "
            f"Make sure the task name is correct."
        )

    try:
        with open(
                os.path.join(dataset_path, task + "_EMM.json"),
                "r",
        ) as json_file:
            task_dict = json.load(json_file)

        train_dataset = CauEegMriMultiModalDataset(
            dataset_path,
            task_dict["train_split"],
            load_event=load_event,
            eeg_file_format=eeg_file_format,
            transform=transform,
        )
        val_dataset = CauEegMriMultiModalDataset(
            dataset_path,
            task_dict["validation_split"],
            load_event=load_event,
            eeg_file_format=eeg_file_format,
            transform=transform,
        )
        test_dataset = CauEegMriMultiModalDataset(
            dataset_path,
            task_dict["test_split"],
            load_event=load_event,
            eeg_file_format=eeg_file_format,
            transform=transform,
        )
    except FileNotFoundError as e:
        print(
            f"ERROR: load_cauemm_task_datasets(dataset_path={dataset_path}) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise
    except ValueError as e:
        print(f"ERROR: load_cauemm_task_datasets(file_format={eeg_file_format}) encounters an error of {e}.")
        raise

    config = {
        k: v
        for k, v in task_dict.items()
        if k
           not in [
               "train_split",
               "validation_split",
               "test_split",
           ]
    }

    if verbose:
        print("task config:")
        pprint.pprint(config, compact=True)
        print("\n", "-" * 100, "\n")

        print("train_dataset[0].keys():")
        pprint.pprint(train_dataset[0].keys(), compact=True)

        if torch.is_tensor(train_dataset[0]):
            print(
                "train signal shape:",
                train_dataset[0]["volume"].shape,
            )
        else:
            print(
                "train signal shape:",
                train_dataset[0]["volume"][0].shape,
            )

        print()
        print("\n" + "-" * 100 + "\n")

        print("val_dataset[0].keys():")
        pprint.pprint(val_dataset[0].keys(), compact=True)
        print("\n" + "-" * 100 + "\n")

        print("test_dataset[0].keys():")
        pprint.pprint(test_dataset[0].keys(), compact=True)
        print("\n" + "-" * 100 + "\n")

    return (
        config,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def load_cauemm_task_split(
        dataset_path: str,
        task: str,
        split: str,
        load_event: bool = True,
        eeg_file_format: str = "memmap",
        transform=None,
        verbose=False,
):
    """Load the CAUMRI dataset for the specified split of the target benchmark task as a PyTorch dataset instance.

    Args:
        dataset_path (str): The file path where the dataset files are located.
        task (str): The target task to load among 'dementia' or 'abnormal'.
        split (str): The desired dataset split to get among "train", "validation", and "test".
        load_event (bool): Whether to load the event information occurred during recording EEG signals.
        eeg_file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Preprocessing process to apply during loading signals.
        verbose (bool): Whether to print the progress during loading the dataset.

    Returns:
        A PyTorch dataset instance for the specified split for the task and their configurations.
    """
    task = task.lower()
    if task not in [
        "abnormal",
        "dementia",
        "abnormal-no-overlap",
        "dementia-no-overlap",
    ]:
        raise ValueError(
            f"load_caumri_task_split(task) receives the invalid task name: {task}. "
            f"Make sure the task name is correct."
        )

    try:
        with open(
                os.path.join(dataset_path, task + "_EMM.json"),
                "r",
        ) as json_file:
            task_dict = json.load(json_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: load_caumri_task_split(dataset_path) encounters an error of {e}. "
            f"Make sure the dataset path is correct."
        )
        raise

    if split in [
        "train",
        "training",
        "train_split",
        "training_split",
    ]:
        dataset = CauEegMriMultiModalDataset(
            dataset_path,
            task_dict["train_split"],
            load_event=load_event,
            eeg_file_format=eeg_file_format,
            transform=transform,
        )
    elif split in [
        "val",
        "validation",
        "val_split",
        "validation_split",
    ]:
        dataset = CauEegMriMultiModalDataset(
            dataset_path,
            task_dict["validation_split"],
            load_event=load_event,
            eeg_file_format=eeg_file_format,
            transform=transform,
        )
    elif split in ["test", "test_split"]:
        dataset = CauEegMriMultiModalDataset(
            dataset_path,
            task_dict["test_split"],
            load_event=load_event,
            eeg_file_format=eeg_file_format,
            transform=transform,
        )
    else:
        raise ValueError(
            f"ERROR: load_cauemm_task_split(split) needs string among of " f"'train', 'validation', and 'test'"
        )

    config = {
        k: v
        for k, v in task_dict.items()
        if k
           not in [
               "train_split",
               "validation_split",
               "test_split",
           ]
    }

    if verbose:
        print(f"{split}_dataset[0].keys():")
        pprint.pprint(dataset[0].keys(), compact=True)

        if torch.is_tensor(dataset[0]):
            print(
                f"{split} volume shape:",
                dataset[0]["volume"].shape,
            )
        else:
            print(
                f"{split} volume shape:",
                dataset[0]["volume"][0].shape,
            )

        print("\n" + "-" * 100 + "\n")

    return config, dataset

def calculate_signal_statistics(
        train_loader,
        preprocess_train=None,
        repeats=5,
        verbose=False,
):
    signal_means = torch.zeros((1,))
    signal_stds = torch.zeros((1,))
    n_count = 0

    for r in range(repeats):
        for i, sample in enumerate(train_loader):
            if preprocess_train is not None:
                preprocess_train(sample)

            signal = sample["signal"]
            std, mean = torch.std_mean(signal, dim=-1, keepdim=True)  # [N, C, L] or [N, (2)C, F, T]

            if r == 0 and i == 0:
                signal_means = torch.zeros_like(mean)
                signal_stds = torch.zeros_like(std)

            signal_means += mean
            signal_stds += std
            n_count += 1

    signal_mean = torch.mean(
        signal_means / n_count,
        dim=0,
        keepdim=True,
    )  # [N, C, L] or [N, (2)C, F, T]
    signal_std = torch.mean(signal_stds / n_count, dim=0, keepdim=True)

    if verbose:
        print("Mean and standard deviation for signal:")
        pprint.pprint(signal_mean, width=250)
        print("-")
        pprint.pprint(signal_std, width=250)
        print("\n" + "-" * 100 + "\n")

    return signal_mean, signal_std


def calculate_age_statistics(train_loader, verbose=False):
    age_means = torch.zeros((1,))
    age_stds = torch.zeros((1,))
    n_count = 0

    for i, sample in enumerate(train_loader):
        age = sample["age"]
        std, mean = torch.std_mean(age, dim=-1, keepdim=True, unbiased=False) # temporaly add unbiased

        if i == 0:
            age_means = torch.zeros_like(mean)
            age_stds = torch.zeros_like(std)

        age_means += mean
        age_stds += std
        n_count += 1

    age_mean = torch.mean(age_means / n_count, dim=0, keepdim=True)
    age_std = torch.mean(age_stds / n_count, dim=0, keepdim=True)

    if verbose:
        print("Age mean and standard deviation:")
        print(age_mean, age_std)
        print("\n" + "-" * 100 + "\n")

    return age_mean, age_std


def calculate_stft_params(
        seq_length,
        n_fft=0,
        hop_ratio=1.0 / 4.0,
        resample_ratio=1.0,
        verbose=False,
):
    if n_fft == 0:
        n_fft = round(math.sqrt(2.0 * seq_length / hop_ratio))
    elif isinstance(n_fft, float):
        n_fft = round(n_fft)
    hop_length = round(n_fft * hop_ratio)
    seq_len_2d = (
        math.floor(n_fft / 2.0) + 1,
        math.floor(seq_length / hop_length * resample_ratio) + 1,
    )

    if verbose:
        print(
            f"Input sequence length: ({seq_length}) would become "
            f"({seq_len_2d[0]}, {seq_len_2d[1]}) "
            f"after the STFT with n_fft ({n_fft}) and hop_length ({hop_length})."
        )
        print("\n" + "-" * 100 + "\n")

    return n_fft, hop_length, seq_len_2d


def compose_transforms(config, verbose=False):
    eeg_transform = []
    mri_transform = []
    eeg_transform_multicrop = []

    #########################
    #   SIGNAL TRANSFORM    #
    #########################
    eeg_transform += [
        EegRandomCrop(
            crop_length=config.get("crop_length", config["seq_length"]),
            length_limit=config.get("signal_length_limit", 10 ** 7),
            multiple=config.get("crop_multiple", 1),
            latency=config.get("latency", 0),
            segment_simulation=config.get("segment_simulation", False),
            return_timing=config.get("crop_timing_analysis", False),
            reject_events=config.get("reject_events", False),
        )
    ]
    eeg_transform_multicrop += [
        EegRandomCrop(
            crop_length=config.get("crop_length", config["seq_length"]),
            length_limit=config.get("signal_length_limit", 10 ** 7),
            multiple=config.get("test_crop_multiple", 8),
            latency=config.get("latency", 0),
            segment_simulation=config.get("segment_simulation", False),
            return_timing=config.get("crop_timing_analysis", False),
            reject_events=config.get("reject_events", False),
        )
    ]

    if config.get("channel_difference", None):
        channel_difference_list = config.get("channel_difference", None)
        if len(channel_difference_list) != 2:
            raise ValueError(f"config['channel_difference'] should be the list of length 2.")
        eeg_transform += [EegChannelDifference(channel_difference_list[0], channel_difference_list[1])]
        eeg_transform_multicrop += [EegChannelDifference(channel_difference_list[0], channel_difference_list[1])]
        signal_header = config["signal_header"]
        config["montage"] = " - ".join([signal_header[i].split("-")[0] for i in channel_difference_list])
    else:
        channel_reduction_list = config.get("channel_reduction_list", [])

        if config.get("EKG", None) not in [
            "O",
            "X",
            None,
        ]:
            raise ValueError(f"config['EKG'] should be one of ['O', 'X', None].")
        elif config.get("EKG", None) == "X":
            channel_reduction_list.append(config["signal_header"].index("EKG"))

        if config.get("photic", None) not in [
            "O",
            "X",
            None,
        ]:
            raise ValueError(f"config['photic'] should be one of ['O', 'X', None].")
        elif config.get("photic", None) == "X":
            channel_reduction_list.append(config["signal_header"].index("Photic"))

        channel_reduction_set = set(channel_reduction_list)

        eeg_transform += [EegDropChannels(sorted([*channel_reduction_set]))]
        eeg_transform_multicrop += [EegDropChannels(sorted([*channel_reduction_set]))]

    ### Numpy to Tensor ###
    eeg_transform += [EegToTensor()]
    eeg_transform_multicrop += [EegToTensor()]

    ### Transform-Compose ###
    eeg_transform = transforms.Compose(eeg_transform)
    eeg_transform_multicrop = transforms.Compose(eeg_transform_multicrop)

    #############################################
    #              IMAGE TRANSFORM              #
    #############################################

    #######################
    ### Pad and Resize  ###
    #######################

    # # Version 1.0 of MRI Transformation #
    resize_size = config.get('mri_resize', 128)
    mri_transform += [MriSpatialPad(256)]
    mri_transform += [MriResize(resize_size)]
    mri_transform += [MriToTensor()]
    mri_transform = transforms.Compose(mri_transform)

    # Version 2.0 of MRI Transformation #
    # resize_size = config.get('mri_resize', 128)
    # mri_transform += [MriDropInvalidRange()]
    # mri_transform += [MriSpatialPad()]
    # mri_transform += [MriResize(resize_size)]
    # mri_transform += [MriToTensor()]
    # mri_transform = transforms.Compose(mri_transform)

    if verbose:
        print("eeg_transform:", eeg_transform)
        print("\n" + "-" * 100 + "\n")

        print(
            "eeg_transform_multicrop:",
            eeg_transform_multicrop,
        )
        print("\n" + "-" * 100 + "\n")
        print()

        print("mri_transform:", eeg_transform)
        print("\n" + "-" * 100 + "\n")
        print()

    return eeg_transform, eeg_transform_multicrop, mri_transform

def compose_preprocess(config, train_loader, verbose=True):
    eeg_preprocess_train = []
    eeg_preprocess_test = []
    mri_preprocess_train = []
    mri_preprocess_test = []

    #####################
    #   EEG PREPROCESS  #
    #####################
    eeg_preprocess_train += [EegToDevice(device=config["device"])]
    eeg_preprocess_test += [EegToDevice(device=config["device"])]

    if "crop_length" not in config:
        config["crop_length"] = config["seq_length"]
    sampling_rate = config.get("sampling_rate", 200)
    config["seq_length"] = math.ceil(config["crop_length"] * config.get("resample", sampling_rate) / sampling_rate)

    if config.get("resample", None):
        eeg_preprocess_train += [EegResample(orig_freq=sampling_rate, new_freq=config["resample"]).to(config["device"])]
        eeg_preprocess_test += [EegResample(orig_freq=sampling_rate, new_freq=config["resample"]).to(config["device"])]

    if "age_mean" not in config or "age_std" not in config:
        (
            config["age_mean"],
            config["age_std"],
        ) = calculate_age_statistics(train_loader, verbose=False)
    eeg_preprocess_train += [
        EegNormalizeAge(
            mean=config["age_mean"],
            std=config["age_std"],
        )
    ]
    eeg_preprocess_test += [
        EegNormalizeAge(
            mean=config["age_mean"],
            std=config["age_std"],
        )
    ]

    if config.get("run_mode", None) == "eval":
        pass
    elif config.get("awgn_age") is None or config["awgn_age"] <= 1e-12:
        pass
    elif config["awgn_age"] > 0.0:
        eeg_preprocess_train += [EegAddGaussianNoiseAge(mean=0.0, std=config["awgn_age"])]
    else:
        raise ValueError(f"config['awgn_age'] have to be None or a positive floating point number")

    ##################################
    # data normalization (1D signal) #
    ##################################
    if config["input_norm"] == "dataset":
        if "signal_mean" not in config or "signal_std" not in config:
            (
                config["signal_mean"],
                config["signal_std"],
            ) = calculate_signal_statistics(
                train_loader,
                repeats=5,
                verbose=False,
            )
        eeg_preprocess_train += [
            EegNormalizeMeanStd(
                mean=config["signal_mean"],
                std=config["signal_std"],
            )
        ]
        eeg_preprocess_test += [
            EegNormalizeMeanStd(
                mean=config["signal_mean"],
                std=config["signal_std"],
            )
        ]
    elif config["input_norm"] == "datapoint":
        eeg_preprocess_train += [EegNormalizePerSignal()]
        eeg_preprocess_test += [EegNormalizePerSignal()]
    elif config["input_norm"] == "no":
        pass
    else:
        raise ValueError(f"config['input_norm'] have to be set to one of ['dataset', 'datapoint', 'no']")

    ###############################
    # dropout channel (1D signal) #
    ###############################
    if config.get("channel_dropout", 0.0) > 1e-8:
        eeg_preprocess_train += [EegChannelDropOut(p=config["channel_dropout"])]

    ##############################################################
    # multiplicative Gaussian noise for augmentation (1D signal) #
    ##############################################################
    if config.get("run_mode", None) == "eval":
        pass
    elif config.get("mgn") is None or config["mgn"] <= 1e-12:
        pass
    elif config["mgn"] > 0.0:
        eeg_preprocess_train += [EegMultiplicativeGaussianNoise(mean=0.0, std=config["mgn"])]
    else:
        raise ValueError(f"config['mgn'] have to be None or a positive floating point number")

    ########################################################
    # additive Gaussian noise for augmentation (1D signal) #
    ########################################################
    if config.get("run_mode", None) == "eval":
        pass
    elif config.get("awgn") is None or config["awgn"] <= 1e-12:
        pass
    elif config["awgn"] > 0.0:
        eeg_preprocess_train += [EegAdditiveGaussianNoise(mean=0.0, std=config["awgn"])]
    else:
        raise ValueError(f"config['awgn'] have to be None or a positive floating point number")

    ###################
    # STFT (1D -> 2D) #
    ###################
    if config.get("eeg_model")["model"].startswith("2D"):
        stft_params = config.pop("stft_params", {})
        (
            n_fft,
            hop_length,
            seq_len_2d,
        ) = calculate_stft_params(
            seq_length=config.get("crop_length", config["seq_length"]),
            n_fft=stft_params.pop("n_fft", 0),
            hop_ratio=stft_params.pop("hop_ratio", 1 / 4.0),
            resample_ratio=config.get("resample", sampling_rate) / sampling_rate,
            verbose=False,
        )
        config["stft_params"] = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            **stft_params,
        }
        config["seq_len_2d"] = seq_len_2d

        eeg_preprocess_train += [EegSpectrogram(**config["stft_params"])]
        eeg_preprocess_test += [EegSpectrogram(**config["stft_params"])]

    ######################################
    ### data normalization (2D signal) ###
    ######################################
    if config.get("eeg_model")["model"].startswith("2D"):
        if config["input_norm"] == "dataset":
            if "signal_2d_mean" not in config or "signal_2d_std" not in config:
                preprocess_temp = transforms.Compose(eeg_preprocess_train)
                preprocess_temp = torch.nn.Sequential(*preprocess_temp.transforms)

                (
                    signal_2d_mean,
                    signal_2d_std,
                ) = calculate_signal_statistics(
                    train_loader,
                    preprocess_train=preprocess_temp,
                    repeats=5,
                    verbose=False,
                )
                config["signal_2d_mean"] = signal_2d_mean
                config["signal_2d_std"] = signal_2d_std

            eeg_preprocess_train += [
                EegNormalizeMeanStd(
                    mean=config["signal_2d_mean"],
                    std=config["signal_2d_std"],
                )
            ]
            eeg_preprocess_test += [
                EegNormalizeMeanStd(
                    mean=config["signal_2d_mean"],
                    std=config["signal_2d_std"],
                )
            ]

        elif config["input_norm"] == "datapoint":
            eeg_preprocess_train += [EegNormalizePerSignal()]
            eeg_preprocess_test += [EegNormalizePerSignal()]

        elif config["input_norm"] == "no":
            pass

        else:
            raise ValueError(f"config['input_norm'] have to be set to one of ['dataset', 'datapoint', 'no']")

    #####################
    #   MRI PREPROCESS  #
    #####################
    mri_preprocess_train += [MriToDevice(device=config["device"])]
    mri_preprocess_test += [MriToDevice(device=config["device"])]

    #######################
    ### Normalization   ###
    #######################
    if config['mri_norm_type'] == 'z_score':
        mri_preprocess_train += [MriNormalize(eps=1e-8, mri_norm_type='z_score')]
        mri_preprocess_test += [MriNormalize(eps=1e-8, mri_norm_type='z_score')]
    elif config['mri_norm_type'] == 'min_max':
        mri_preprocess_train += [MriNormalize(eps=1e-8, mri_norm_type='min_max')]
        mri_preprocess_test += [MriNormalize(eps=1e-8, mri_norm_type='min_max')]
    elif config['mri_norm_type'] == 'no':
        pass
    else:
        raise ValueError(f"config['mri_norm_type'] have to be set to one of ['z_score', 'min_max', 'no']")

    #######################
    # Compose All at Once #
    #######################
    eeg_preprocess_train = transforms.Compose(eeg_preprocess_train)
    eeg_preprocess_train = torch.nn.Sequential(*eeg_preprocess_train.transforms)

    eeg_preprocess_test = transforms.Compose(eeg_preprocess_test)
    eeg_preprocess_test = torch.nn.Sequential(*eeg_preprocess_test.transforms)

    mri_preprocess_train = transforms.Compose(mri_preprocess_train)
    mri_preprocess_train = torch.nn.Sequential(*mri_preprocess_train.transforms)

    mri_preprocess_test = transforms.Compose(mri_preprocess_test)
    mri_preprocess_test = torch.nn.Sequential(*mri_preprocess_test.transforms)

    preprocess_train = (eeg_preprocess_train, mri_preprocess_train)
    preprocess_test = (eeg_preprocess_test, mri_preprocess_test)

    if verbose:
        print("eeg_preprocess_train:", eeg_preprocess_train)
        print("\n" + "-" * 100 + "\n")

        print("eeg_preprocess_test:", eeg_preprocess_test)
        print("\n" + "-" * 100 + "\n")

        print("mri_preprocess_train:", mri_preprocess_train)
        print("\n" + "-" * 100 + "\n")

        print("mri_preprocess_test:", mri_preprocess_test)
        print("\n" + "-" * 100 + "\n")

    return preprocess_train, preprocess_test


def make_cauemm_dataloader(
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        multicrop_test_dataset,
):
    if config["device"] == "cpu":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = config.get("num_workers", 0)  # A number other than 0 can cause an error
        pin_memory = True

    batch_size = config["minibatch"] / config.get("crop_multiple", 1)
    if batch_size < 1 or batch_size % 1 > 1e-12:
        raise ValueError(
            f"ERROR: config['minibatch']={config['minibatch']} "
            f"is not multiple of config['crop_multiple']={config['crop_multiple']}."
        )
    batch_size = round(batch_size)

    multi_batch_size = config["minibatch"] / config.get("test_crop_multiple", 1)
    if multi_batch_size < 1 or multi_batch_size % 1 > 1e-12:
        raise ValueError(
            f"ERROR: config['minibatch']={config['minibatch']} "
            f"is not multiple of config['test_crop_multiple']={config['test_crop_multiple']}."
        )
    config["multi_batch_size"] = round(multi_batch_size)

    if config.get("ddp", False):
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    if config.get("run_mode", None) == "train":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=emm_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=emm_collate_fn,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=emm_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=emm_collate_fn,
    )

    multicrop_test_loader = DataLoader(
        multicrop_test_dataset,
        batch_size=config["multi_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=emm_collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        multicrop_test_loader,
    )

def build_emm_dataset_for_train(config, verbose=False):
    dataset_path = config["dataset_path"]
    if "cwd" in config:
        dataset_path = os.path.join(config["cwd"], dataset_path)

    config_dataset = load_cauemm_config(dataset_path)
    config.update(**config_dataset)

    if "run_mode" not in config.keys():
        print("\n" + "=" * 80 + "\n")
        print('WARNING: run_mode is not specified.\n \t==> run_mode is set to "train" automatically.')
        print("\n" + "=" * 80 + "\n")
        config["run_mode"] = "train"

    (eeg_transform, eeg_transform_multicrop, mri_transform) = compose_transforms(config, verbose=verbose)
    config["eeg_transform"] = eeg_transform
    config["eeg_transform_multicrop"] = eeg_transform_multicrop
    config["mri_transform"] = mri_transform
    load_event = config["load_event"] or config.get("reject_events", False)

    (
        config_task,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = load_cauemm_task_datasets(
        dataset_path=dataset_path,
        task=config["task"],
        load_event=load_event,
        eeg_file_format=config["eeg_file_format"],
        transform=(eeg_transform,mri_transform),
        verbose=verbose,
    )
    config.update(**config_task)

    (
        _,
        multicrop_test_dataset,
    ) = load_cauemm_task_split(
        dataset_path=dataset_path,
        task=config["task"],
        split="test",
        load_event=load_event,
        eeg_file_format=config["eeg_file_format"],
        transform=(eeg_transform_multicrop, mri_transform),
        verbose=verbose,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        multicrop_test_loader,
    ) = make_cauemm_dataloader(
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        multicrop_test_dataset,
    )
    (
        preprocess_train,
        preprocess_test,
    ) = compose_preprocess(config, train_loader, verbose=verbose)
    config["preprocess_train"] = preprocess_train
    config["preprocess_test"] = preprocess_test
    config["eeg_model"]["in_channels"] = preprocess_train[0](next(iter(train_loader)))["signal"].shape[1]
    config["out_dims"] = len(config["class_label_to_name"])

    return (
        train_loader,
        val_loader,
        test_loader,
        multicrop_test_loader,
    )