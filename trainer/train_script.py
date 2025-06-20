import os
from copy import deepcopy
import torch
import wandb
import pprint

from .train_core import train_multistep, ssl_train_multistep
from .utils import wandb_config_update
from datetime import datetime
from optim import get_optimizer
from optim import get_lr_scheduler
from .evaluate import check_accuracy
from .evaluate import check_accuracy_extended
from .evaluate import check_accuracy_multicrop
from .visualize import draw_roc_curve, draw_confusion


def train_script(config, model, train_loader, val_loader, test_loader, multicrop_test_loader, preprocess_train, preprocess_test):
    # only the main process of DDP logs, evaluates, and saves
    main_process = config["ddp"] is False or config["device"].index == 0

    if main_process:
        print(f"\n{'*'*30} {'Configurations for Train':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    # load if using an existing model
    if config.get("init_from", None):
        init_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["init_from"]}/')
        checkpoint = torch.load(os.path.join(init_path, "checkpoint.pt"), map_location=config["device"])
        model.load_state_dict(checkpoint["model_state"])
        pprint.pprint(f'Load an existing model from {config["init_from"]}\n', width=120)

    # wandb init
    if main_process and config["use_wandb"]:
        if config.get("resume", None) is None:
            wandb.init(project=config.get("project", "CAUEMM"), reinit=True)
            wandb.run.name = wandb.run.id
        else:
            wandb.init(
                project=config.get("project", "CAUEMM"),
                id=config["resume"],
                resume="must",
            )

    # training iteration and other conditions
    config["base_lr"] = config["base_lr"] * config.get("search_multiplier", 1.0)
    config["iterations"] = round(config["total_samples"] / config["minibatch"] / config.get("ddp_size", 1))
    config["warmup_steps"] = max(round(config["iterations"] * config["warmup_ratio"]), config["warmup_min"])
    history_interval = max(config["iterations"] // config["num_history"], 1)

    # generate the trainers
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(
        optimizer,
        config["lr_scheduler_type"],
        iterations=config["iterations"],
        warmup_steps=config["warmup_steps"],
    )
    amp_scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", False) else None

    # local variable for training loop
    best_val_acc = 0
    best_model_state = deepcopy(model.state_dict())
    i_step = 0

    # load if resuming
    if config.get("resume", None):
        resume = config["resume"]
        save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["resume"]}/')
        checkpoint = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location=config["device"])
        best_model_state = checkpoint["model_state"]
        model.load_state_dict(best_model_state)
        best_val_acc = check_accuracy(
            model,
            val_loader,
            preprocess_test,
            config,
            repeat=config.get("test_accuracy_repeat", 30),
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        config = checkpoint["config"]
        if main_process and config["use_wandb"]:
            wandb_config_update(config, allow_val_change=True)
        i_step = checkpoint["optimizer_state"]["state"][0]["step"]
        if not isinstance(i_step, int):
            i_step = int(i_step.detach().cpu().numpy())
        print(f"\n{'*'*30} {f'Training resumes from {resume}':^30} {'*'*30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*'*92}\n")

    if main_process:
        # update configurations
        if config["use_wandb"]:
            wandb_config_update(config)

            # track gradients and weights statistics if needed
            if config.get("watch_model", False):
                wandb.watch(model, log="all", log_freq=history_interval, log_graph=True)

        # directory to save
        run_name = wandb.run.name if config["use_wandb"] else datetime.now().strftime("%Y-%m%d-%H%M")
        if config["save_model"]:
            save_path = os.path.join(config.get("cwd", ""), f"local/checkpoint/{run_name}/")
            os.makedirs(save_path, exist_ok=True)

    # train and validation routine
    while i_step < config["iterations"]:
        i_step += history_interval

        # train during 'history_interval' steps
        loss, train_acc = train_multistep(
            model=model,
            loader=train_loader,
            preprocess=preprocess_train,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=amp_scaler,
            config=config,
            steps=history_interval,
        )
        # validation accuracy
        val_acc = check_accuracy(
            model,
            val_loader,
            preprocess_test,
            config,
            repeat=config.get("val_accuracy_repeat", 30),
        )

        # log
        if main_process:
            if config["use_wandb"]:
                wandb.log(
                    {
                        "Loss": loss,
                        "Train Accuracy": train_acc,
                        "Validation Accuracy": val_acc,
                        "Learning Rate": optimizer.state_dict()["param_groups"][0]["lr"],
                    },
                    step=i_step * config["minibatch"],
                )
            else:
                print(
                    f"{i_step:>8} / {config['iterations']:>8} iter - "
                    f"Loss: {loss:.4f}, Train Acc.: {train_acc:.4f}, Val. Acc.: {val_acc:.4f}"
                )

            # save the model
            if config["save_model"]:
                checkpoint = {
                    "model_state": model.state_dict(),
                    "config": config,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_path, "checkpoint_.pt"))
                os.replace(
                    os.path.join(save_path, "checkpoint_.pt"),
                    os.path.join(save_path, "checkpoint.pt"),
                )

            # save the best model so far
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())

    # calculate the test accuracy for best and last models
    if main_process:
        last_model_state = deepcopy(model.state_dict())
        last_test_result = check_accuracy_extended(
            model=model,
            loader=test_loader,
            preprocess=preprocess_test,
            config=config,
            repeat=config.get("test_accuracy_repeat", 30),
        )
        last_test_acc = last_test_result[0]

        model.load_state_dict(best_model_state)
        best_test_result = check_accuracy_extended(
            model=model,
            loader=test_loader,
            preprocess=preprocess_test,
            config=config,
            repeat=config.get("test_accuracy_repeat", 30),
        )
        best_test_acc = best_test_result[0]

        if last_test_acc < best_test_acc:
            model_state = best_model_state
            test_result = best_test_result
        else:
            model_state = last_model_state
            test_result = last_test_result

        model.load_state_dict(model_state)
        test_acc, score, target, test_confusion, throughput = test_result

        # calculate the test accuracy of the final model using multiple crop averaging
        multicrop_test_acc = check_accuracy_multicrop(
            model=model,
            loader=multicrop_test_loader,
            preprocess=preprocess_test,
            config=config,
            repeat=config.get("test_accuracy_repeat", 30),
        )

        # save the model
        if config["save_model"]:
            checkpoint = {
                "model_state": model_state,
                "config": config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
            os.replace(
                os.path.join(save_path, "checkpoint.pt"),
                os.path.join(save_path, "checkpoint.pt"),
            )

        # leave the message
        if config["use_wandb"]:
            wandb.log(
                {
                    "Test Accuracy": test_acc,
                    "(Best, Last) Test Accuracy": (
                        "Best" if last_test_acc < best_test_acc else "Last",
                        round(best_test_acc, 2),
                        round(last_test_acc, 2),
                    ),
                    "Confusion Matrix (Array)": test_confusion,
                    "Multi-Crop Test Accuracy": multicrop_test_acc,
                }
            )
        else:
            print(f"\n{'*'*30} {run_name:^30} {'*'*30}\n")
            pprint.pprint(
                {
                    f"Test Accuracy": test_acc,
                    "(Best, Last) Test Accuracy": (
                        "Best" if last_test_acc < best_test_acc else "Last",
                        round(best_test_acc, 2),
                        round(last_test_acc, 2),
                    ),
                    "Confusion Matrix (Array)": test_confusion,
                    "Multi-Crop Test Accuracy": multicrop_test_acc,
                }
            )
            print(f"\n{'*'*92}\n")

        if config["draw_result"]:
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

        if config["use_wandb"]:
            wandb.run.finish()

        del last_model_state

    # release memory
    del optimizer, scheduler, best_model_state
    return

def train_ssl_script(config, model, loader, preprocess):
    # only the main process of DDP logs, evaluates, and saves
    main_process = config["ddp"] is False or config["device"].index == 0

    if main_process:
        print(f"\n{'*' * 30} {'Configurations for Train':^30} {'*' * 30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*' * 92}\n")

    # load if using an existing model
    if config.get("init_from", None):
        init_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["init_from"]}/')
        checkpoint = torch.load(os.path.join(init_path, "checkpoint.pt"), map_location=config["device"])
        model.load_state_dict(checkpoint["ssl_model_state"])
        pprint.pprint(f'Load an existing model from {config["init_from"]}\n', width=120)

    # wandb init
    if main_process and config["use_wandb"]:
        if config.get("resume", None) is None:
            wandb.init(project=config.get("project", "noname"), reinit=True)
            wandb.run.name = wandb.run.id
        else:
            wandb.init(
                project=config.get("project", "noname"),
                id=config["resume"],
                resume="must",
            )


    # training iteration and other conditions
    config["base_lr"] = config["base_lr"] * config.get("search_multiplier", 1.0)
    config["iterations"] = round(config["total_samples"] / config["minibatch"] / config.get("ddp_size", 1))
    config["warmup_steps"] = max(round(config["iterations"] * config["warmup_ratio"]), config["warmup_min"])
    history_interval = max(config["iterations"] // config["num_history"], 1)

    # generate the trainers
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(
        optimizer,
        config["lr_scheduler_type"],
        iterations=config["iterations"],
        warmup_steps=config["warmup_steps"],
    )
    amp_scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", False) else None

    # local variable for training loop
    i_step = 0

    # load if resuming
    if config.get("resume", None):
        resume = config["resume"]
        save_path = os.path.join(config.get("cwd", ""), f'local/checkpoint/{config["resume"]}/')
        checkpoint = torch.load(os.path.join(save_path, "checkpoint.pt"), map_location=config["device"])
        model.load_state_dict(checkpoint["ssl_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        config = checkpoint["config"]
        if main_process and config["use_wandb"]:
            wandb.config.update(config, allow_val_change=True)
        i_step = checkpoint["optimizer_state"]["state"][0]["step"]
        if not isinstance(i_step, int):
            i_step = int(i_step.detach().cpu().numpy())
        print(f"\n{'*' * 30} {f'Training resumes from {resume}':^30} {'*' * 30}\n")
        pprint.pprint(config, width=120)
        print(f"\n{'*' * 92}\n")

    if main_process:
        # update configurations
        if config["use_wandb"]:
            wandb.config.update(config)

            # track gradients and weights statistics if needed
            if config.get("watch_model", False):
                wandb.watch(model, log="all", log_freq=history_interval, log_graph=True)

        # directory to save
        run_name = wandb.run.name if config["use_wandb"] else datetime.now().strftime("%Y-%m%d-%H%M")
        if config["save_model"]:
            save_path = os.path.join(config.get("cwd", ""), f"local/checkpoint/{run_name}/")
            os.makedirs(save_path, exist_ok=True)

    # train and validation routine
    while i_step < config["iterations"]:
        i_step += history_interval

        # train during 'history_interval' steps
        loss = ssl_train_multistep(
            model=model,
            loader=loader,
            preprocess=preprocess,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=amp_scaler,
            config=config,
            steps=history_interval,
        )
        # log
        if main_process:
            if config["use_wandb"]:
                wandb.log(
                    {
                        "Loss": loss,
                        "Learning Rate": optimizer.state_dict()["param_groups"][0]["lr"],
                    },
                    step=i_step * config["minibatch"],
                )
            else:
                print(f"{i_step: >8} / {config['iterations']: >8} iter - Loss: {loss:.4f}")

            # save the model
            if config["save_model"]:
                checkpoint = {
                    "ssl_model_state": model.state_dict(),
                    "config": config,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_path, "checkpoint_.pt"))
                os.replace(
                    os.path.join(save_path, "checkpoint_.pt"),
                    os.path.join(save_path, "checkpoint.pt"),
                )

    # calculate the test accuracy for best and last models
    if main_process:
        # save the model
        if config["save_model"]:
            checkpoint = {
                "ssl_model_state": model.state_dict(),
                "config": config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
            os.replace(
                os.path.join(save_path, "checkpoint.pt"),
                os.path.join(save_path, "checkpoint.pt"),
            )

        # leave the message
        if config["use_wandb"]:
            wandb.run.finish()

    # release memory
    del optimizer, scheduler
    return

