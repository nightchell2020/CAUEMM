import gc
import pprint
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from datasets.cauemm_script import build_emm_dataset_for_train
from train import load_pretrained_params, generate_model, check_device_env
from copy import deepcopy
from trainer.evaluate import check_accuracy_extended, check_accuracy_multicrop
from trainer.visualize import draw_roc_curve, draw_confusion


def compose_dataset(config):
    return build_emm_dataset_for_train(config)


def inference_script(
        config,
        model,
        test_loader,
        multicrop_test_loader,
        preprocess_test,
    ):
    model_state = deepcopy(model.state_dict())
    model.load_state_dict(model_state)
    test_result = check_accuracy_extended(
        model=model,
        loader=test_loader,
        preprocess=preprocess_test,
        config=config,
        repeat=config.get("test_accuracy_repeat", 30),
    )
    test_acc, score, target, test_confusion, throughput = test_result
    multicrop_test_acc = check_accuracy_multicrop(
        model=model,
        loader=multicrop_test_loader,
        preprocess=preprocess_test,
        config=config,
        repeat=config.get("test_accuracy_repeat", 30),
    )

    pprint.pprint(
        {
            f"Test Accuracy": test_acc,
            "Confusion Matrix (Array)": test_confusion,
            "Multi-Crop Test Accuracy": multicrop_test_acc,
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


@hydra.main(config_path="configs", config_name="infer_config")
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