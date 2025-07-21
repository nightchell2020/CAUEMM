import os
import gc
import torch
import hydra
import numpy as np
import torch.multiprocessing as mp

from copy import deepcopy
from collections import OrderedDict
from packaging import version
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.cauemm_script import build_emm_dataset_for_train
from models.utils import count_parameters, get_model_size
from trainer.train_script import train_script
from trainer.utils import merge_state_dicts, add_prefix_to_pretrained_weights, print_gpu_utilization

HYDRA_FULL_ERROR=1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def check_device_env(config):
    if not torch.cuda.is_available():
        raise ValueError("ERROR: No GPU is available. Check the environment again!!")

    # assign GPU
    config["device"] = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    device_name = torch.cuda.get_device_name(0)
    # minibatch sizes
    if "minibatch" not in config:
        # set the minibatch size according to the GPU memory
        if "3090" in device_name:
            config["minibatch"] = config["minibatch_4090"] // 2
        elif "2080" in device_name:
            config["minibatch"] = config["minibatch_4090"] // 4
        elif "1080" in device_name:
            config["minibatch"] = config["minibatch_4090"] // 8
        elif "1070" in device_name:
            config["minibatch"] = config["minibatch_4090"] // 8
        elif "4090" in device_name:
            config["minibatch"] = config["minibatch_4090"]
        else:
            config["minibatch"] = 128
            print("*" * 150)
            print(
                f"- WARNING: this process set the minibatch size as {config['minibatch']}, "
                f"assuming that your VRAM size of GPU is equivalent to NVIDIA RTX 4090."
            )
            print(f"- If you want to change the minibatch size, add '++minibatch=MINIBACH_SIZE' option to the command.")
            print("*" * 150)

    # distributed training
    if config.get("ddp", False):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            config["ddp_size"] = config.get("ddp_size", world_size)
        else:
            raise ValueError(
                f"ERROR: There are not sufficient GPUs to launch the DDP training: {world_size}. "
                f"Check the environment again!!"
            )

def initialize_ddp(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    config = deepcopy(config)
    config["device"] = torch.device(f"cuda:{rank}")
    return config

def set_seed(config, rank):
    if config.get("seed", 0) >= 0:
        seed = config.get("seed", 0)
        seed = seed + rank if rank is not None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)

def compose_dataset(config):
    return build_emm_dataset_for_train(config)


def generate_model(config):
    model = hydra.utils.instantiate(config)
    get_model_size(model)
    print("----------------------------------------------------")
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Using CPU")
    print("----------------------------------------------------")
    if config.get("ddp", False):
        torch.cuda.set_device(config["device"])
        model.cuda(config["device"])
        model = DDP(model, device_ids=[config["device"]], find_unused_parameters=True)
        config["output_length"] = model.module.get_output_length()
        config["num_params"] = count_parameters(model)
        torch.distributed.barrier()
    else:
        model = model.to(config["device"])
        config["output_length"] = model.get_output_length()
        config["num_params"] = count_parameters(model)

    if "model_compile" in config.keys():
        if version.parse("2.0.0") <= version.parse(torch.__version__):
            model = torch.compile(model, mode=config.get("model_compile", None))
        else:
            print(
                "WARNING: PyTorch version ({str(torch.__version__)}) older than 2.0.0 cannot compile the model. "
                "The config option['model_compile'] is ignored."
            )
            config.pop("model_compile", None)

    return model


def load_pretrained_params(model, config):
    # model_state_checker = model.state_dict()
    eeg_weight = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["eeg_model"]["load_pretrained"]}/')
    eeg_ckpt = torch.load(os.path.join(eeg_weight, "checkpoint.pt"), map_location=config["device"])
    eeg_state = add_prefix_to_pretrained_weights(eeg_ckpt['model_state'], "eeg_model")

    mri_weight = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["mri_model"]["load_pretrained"]}/')
    mri_ckpt = torch.load(os.path.join(mri_weight, "checkpoint.pth"), map_location=config["device"])["state_dict"]



    if eeg_ckpt["config"]["ddp"] == config["ddp"]:  # Both are DDP
        ckpt = merge_state_dicts(eeg_state, mri_ckpt)
        model.load_state_dict(ckpt, strict=False)
    elif eeg_ckpt["config"]["ddp"]:                 # Only pretrained are DDP
        eeg_model_state_ddp = deepcopy(eeg_ckpt["model_state"])
        eeg_model_state = OrderedDict()
        for k, v in eeg_model_state_ddp.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            eeg_model_state[name] = v
        ckpt = merge_state_dicts(eeg_model_state, mri_ckpt)
        model.load_state_dict(ckpt, strict=False)
    else:                                           # Pretrained are not DDP
        ckpt = merge_state_dicts(eeg_state, mri_ckpt)
        model.module.load_state_dict(ckpt, strict=False)


def prepare_and_run_train(rank, world_size, config):
    # collect some garbage
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # fix the seed for reproducibility (a negative seed value means not fixing)
    set_seed(config, rank)

    # setup for distributed training
    if config.get("ddp", False):
        config = initialize_ddp(rank, world_size, config)

    # compose dataset
    train_loader, val_loader, test_loader, multicrop_test_loader = compose_dataset(config)

    # generate the model and update some configurations
    model = generate_model(config)
    print_gpu_utilization()

    # load pretrained model if needed
    if "load_pretrained" in config.keys():
        load_pretrained_params(model, config)

    # train
    train_script(
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        multicrop_test_loader,
        config["preprocess_train"],
        config["preprocess_test"],
    )

    # cleanup
    if config.get("ddp", False):
        torch.distributed.destroy_process_group()


@hydra.main(config_path="config", config_name="default")
def my_app(cfg: DictConfig) -> None:
    # initialize the configurations
    config = {
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.trainer),
        **OmegaConf.to_container(cfg.model),
        "cwd": HydraConfig.get().runtime.cwd,
    }

    # check the workstation environment and update some configurations
    check_device_env(config)

    # build the dataset and train the model
    if config.get("ddp", False):
        mp.spawn(
            prepare_and_run_train,
            args=(config["ddp_size"], config),
            nprocs=config["ddp_size"],
            join=True,
        )
    else:
        prepare_and_run_train(rank=None, world_size=None, config=config)


if __name__ == "__main__":
    my_app()