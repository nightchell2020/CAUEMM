import json
import math
import os
import pprint

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .eeg_pipeline import eeg_collate_fn

from .cauemm_dataset import CauEegMriMultiModalDataset

def load_cauemm_config(dataset_path: str):
    """Load the configuration of the CAUMRI dataset.

    Args:
        dataset_path (str): The file path where the dataset files are located.
    """
    try:
        with open(
                os.path.join(dataset_path, "annotation.json"),
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
        transform=None,
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
                os.path.join(dataset_path, "annotation.json"),
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
                os.path.join(dataset_path, task + ".json"),
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
                os.path.join(dataset_path, task + ".json"),
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

#ToDo : transforms...
def compose_transforms(config, verbose=False):
    transform = []
    transform_multicrop = []

    return transform, transform_multicrop

def make_cauemm_dataloader(
        config,
        train_dataset,
        val_dataset,
        test_dataset,
        multicrop_test_dataset,
        verbose=False
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
            collate_fn=eeg_collate_fn,
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
            collate_fn=eeg_collate_fn,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn,
    )

    multicrop_test_loader = DataLoader(
        multicrop_test_dataset,
        batch_size=config["multi_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn,
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

    (transform, transform_multicrop) = compose_transforms(config, verbose=verbose)
    config["transform"] = transform
    config["transform_multicrop"] = transform_multicrop
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
        transform=transform,
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
        transform=transform_multicrop,
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
        verbose=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        multicrop_test_loader,
    )