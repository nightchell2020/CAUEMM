import time
import wandb
import re
import torch
import torch.nn.functional as F

from collections import OrderedDict
from torchvision.transforms import Compose
from torch.nn import Sequential
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def load_sweep_config(config):
    # load default configurations not selected by wandb.sweep
    cfg_sweep = dict()
    for k, v in config.items():
        if k not in [wandb_key.split(".")[-1] for wandb_key in wandb.config.keys()]:
            cfg_sweep[k] = v

    # load the selected configurations from wandb sweep with preventing callables from type-conversion to str
    for k, v in wandb.config.items():
        k = k.split(".")[-1]
        if k not in cfg_sweep:
            cfg_sweep[k] = v

    return cfg_sweep


def wandb_config_update(config, allow_val_change=True):
    config_update = {}
    for k, v in config.items():
        if isinstance(v, (Compose, Sequential)):
            config_update[k] = v.__repr__().splitlines()
        else:
            config_update[k] = v
    wandb.config.update(config_update, allow_val_change=allow_val_change)


class TimeElapsed(object):
    def __init__(self, header=""):
        self.header = header
        self.counter = 1
        self.start = time.time()

    def restart(self):
        self.start = time.time()
        self.counter = 1

    def elapsed_str(self):
        end = time.time()
        time_str = f"{self.counter:3d}> {end - self.start :.5f}"
        self.start = end
        self.counter += 1
        return time_str

def add_prefix_to_pretrained_weights(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        new_key = f"{prefix}.{k}"
        new_state_dict[new_key] = v
    return new_state_dict


def edit_MedicalNet_pretrained(state_dict):
    def convert_key(key):
        # conv1 → input_stage.0
        if key.startswith("module.conv1"):
            return key.replace("module.conv1", "module.input_stage.0")
        if key.startswith("module.bn1"):
            return key.replace("module.bn1", "module.input_stage.1")

        # layerX.Y.convZ → conv_stageX.Y.convZ
        m = re.match(r"module\.layer(\d+)\.(\d+)\.conv(\d+)(.*)", key)
        if m:
            x, y, z, tail = m.groups()
            # if x != '4':
            #     y = str(int(y) + 1)
            return f"module.conv_stage{x}.{y}.conv{z}{tail}"

        # layerX.Y.bnZ → conv_stageX.Y.normZ
        m = re.match(r"module\.layer(\d+)\.(\d+)\.bn(\d+)(.*)", key)
        if m:
            x, y, z, tail = m.groups()
            # if x != '4':
            #     y = str(int(y) + 1)
            return f"module.conv_stage{x}.{y}.norm{z}{tail}"

        # layerX.Y.downsample.X → conv_stageX.Y.downsample.X
        m = re.match(r"module\.layer(\d+)\.(\d+)\.downsample\.(\d+)(.*)", key)
        if m:
            x, y, d, tail = m.groups()
            # if x != '4':
            #     y = str(int(y) + 1)
            return f"module.conv_stage{x}.{y}.downsample.{d}{tail}"
        return key

    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_key = convert_key(k)
        new_state[new_key] = v
    return new_state

# def edit_MedicalNet_pretrained(state_dict):
#     def convert_key(key):
#         # conv1 → input_stage.0
#         if key.startswith("module.conv1"):
#             return key.replace("module.conv1", "mri_model.input_stage.0")
#         if key.startswith("module.bn1"):
#             return key.replace("module.bn1", "mri_model.input_stage.1")
#
#         # layerX.Y.convZ → conv_stageX.Y.convZ
#         m = re.match(r"module\.layer(\d+)\.(\d+)\.conv(\d+)(.*)", key)
#         if m:
#             x, y, z, tail = m.groups()
#             if x != '4':
#                 y = str(int(y) + 1)
#             return f"mri_model.conv_stage{x}.{y}.conv{z}{tail}"
#
#         # layerX.Y.bnZ → conv_stageX.Y.normZ
#         m = re.match(r"module\.layer(\d+)\.(\d+)\.bn(\d+)(.*)", key)
#         if m:
#             x, y, z, tail = m.groups()
#             if x != '4':
#                 y = str(int(y) + 1)
#             return f"mri_model.conv_stage{x}.{y}.norm{z}{tail}"
#
#         # layerX.Y.downsample.X → conv_stageX.Y.downsample.X
#         m = re.match(r"module\.layer(\d+)\.(\d+)\.downsample\.(\d+)(.*)", key)
#         if m:
#             x, y, d, tail = m.groups()
#             if x != '4':
#                 y = str(int(y) + 1)
#             return f"mri_model.conv_stage{x}.{y}.downsample.{d}{tail}"
#         return key
#
#     new_state = OrderedDict()
#     for k, v in state_dict.items():
#         new_key = convert_key(k)
#         new_state[new_key] = v
#     return new_state


def merge_state_dicts(dict1, dict2, overwirte=True):
    merged = OrderedDict()

    for k,v in dict1.items():
        merged[k] = v

    for k,v in dict2.items():
        if k in merged and not overwirte:
            continue
        merged[k] = v

    return merged