import gc
import os
import pprint
import torch
import hydra
from medcam import medcam
from torch.amp import autocast
from hydra.core.hydra_config import HydraConfig
from collections import OrderedDict
from trainer.mixup_util import mixup_data, mixup_criterion
from omegaconf import DictConfig, OmegaConf
from datasets.cauemm_script import build_emm_dataset_for_train
from train import generate_model, check_device_env
from copy import deepcopy
from trainer.evaluate import check_accuracy_extended, check_accuracy_multicrop
from trainer.visualize import draw_roc_curve, draw_confusion


def compose_dataset(config):
    return build_emm_dataset_for_train(config)

def load_pretrained_params(model, config):
    save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["load_pretrained"]}/')
    ckpt = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location=config["device"])

    if ckpt["config"]["ddp"] == config["ddp"]:
        model.load_state_dict(ckpt["model_state"])
    elif ckpt["config"]["ddp"]:
        model_state_ddp = deepcopy(ckpt["model_state"])
        model_state = OrderedDict()
        for k, v in model_state_ddp.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            model_state[name] = v
        model.load_state_dict(model_state)
    else:
        model.module.load_state_dict(ckpt["model_state"])



def inference_script(
        config,
        model,
        test_loader,
        multicrop_test_loader,
        preprocess,
    ):
    model_state = deepcopy(model.state_dict())
    model.load_state_dict(model_state)


    model.eval()
    for sample_batched in test_loader:
        # preprocessing (this includes to-device operation)
        preprocess[0](sample_batched)
        preprocess[1](sample_batched)

        # pull the data
        signal = sample_batched["signal"]
        volume = sample_batched["volume"]
        age = sample_batched["age"]
        y = sample_batched["class_label"]
        signal, age, y1, y2, lam, mixup_index = mixup_data(signal, age, y, config["mixup"], config["device"])

        with autocast('cuda', enabled=config.get("mixed_precision", False)):
            # forward pass
            output = model([signal, volume, age])
        print("## I am Here ! ##")

    test_result = check_accuracy_extended(
        model=model,
        loader=test_loader,
        preprocess=preprocess,
        config=config,
        repeat=config.get("test_accuracy_repeat", 30),
    )
    test_acc, score, target, test_confusion, throughput, precision, recall, f1_score = test_result
    multicrop_test_acc = check_accuracy_multicrop(
        model=model,
        loader=multicrop_test_loader,
        preprocess=preprocess,
        config=config,
        repeat=config.get("test_accuracy_repeat", 30),
    )

    pprint.pprint(
        {
            f"Test Accuracy": test_acc,
            "Confusion Matrix (Array)": test_confusion,
            "Multi-Crop Test Accuracy": multicrop_test_acc,
            "Precision": precision,
            "Recall": recall,
            "f1 score": f1_score,
        }
    )
    print(f"\n{'*' * 92}\n")

    draw_roc_curve(
        score,
        target,
        config["class_label_to_name"],
        use_wandb=config["use_wandb"],
    )
    draw_confusion(
        test_confusion,
        config["class_label_to_name"],
        use_wandb=config["use_wandb"],
    )


def run_inference(config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    _, _, test_loader, multicrop_test_loader = compose_dataset(config)
    model = generate_model(config)
    load_pretrained_params(model, config)

    inference_script(config, model, test_loader, multicrop_test_loader, config["preprocess_test"])


@hydra.main(config_path="config", config_name="infer_config")
def my_app(cfg: DictConfig) -> None:
    config = {
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.trainer),
        **OmegaConf.to_container(cfg.model),
        "cwd": HydraConfig.get().runtime.cwd,
    }

    # check the workstation environment and update some configurations
    check_device_env(config)

    run_inference(config=config)

if __name__ == "__main__":
    my_app()