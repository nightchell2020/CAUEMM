
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torchvision.ops as ops
from .mixup_util import mixup_data, mixup_criterion




def train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()
    if config["eeg_freeze"]:
        for param in model.module.eeg_model.parameters():
            param.requires_grad = False
    if config["mri_freeze"]:
        for param in model.module.mri_model.parameters():
            param.requires_grad = False

    # init
    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess[0](sample_batched)
            preprocess[1](sample_batched)

            # pull the data
            signal = sample_batched["signal"]
            volume = sample_batched["volume"]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # mix_up the mini-batched data
            signal, age, y1, y2, lam, mixup_index = mixup_data(signal, age, y, config["mixup"], config["device"])

            # mixed precision training if needed
            with autocast('cuda', enabled=config.get("mixed_precision", False)):
                # forward pass
                output = model([signal, volume, age])
                if isinstance(output, tuple):
                    output, output_kd = output
                else:
                    output_kd = output

                if config['eeg_model']["use_age"] == "estimate":
                    output_age = output[:, -1]
                    output = output[:, :-1]

                ####################################### Loss Function ####################################################

                if config["criterion"] == "cross-entropy":
                    loss = mixup_criterion(F.cross_entropy, output, y1, y2, lam)
                elif config["criterion"] == "multi-bce":
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))
                    loss = mixup_criterion(
                        F.binary_cross_entropy_with_logits,
                        output,
                        y1_oh.float(),
                        y2_oh.float(),
                        lam,
                    )
                elif config["criterion"] == "svm":
                    loss = mixup_criterion(F.multi_margin_loss, output, y1, y2, lam)
                elif config["criterion"] == 'focal':
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))

                    loss = mixup_criterion(ops.focal_loss.sigmoid_focal_loss, output, y1_oh.float(), y2_oh.float(), lam, reduction='mean')
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = output.argmax(dim=-1)
            correct1 = pred.squeeze().eq(y1).sum().item()
            correct2 = pred.squeeze().eq(y2).sum().item()
            correct += lam * correct1 + (1.0 - lam) * correct2
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc

def unimodal_train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    # init
    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # pull the data
            volume = sample_batched["volume"]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # mix_up the mini-batched data
            # but we do not use mix-up aug b/c MRI data, so config['mixup'] set to be 0
            volume, age, y1, y2, lam, mixup_index = mixup_data(volume, age, y, config["mixup"], config["device"])

            # mixed precision training if needed
            with autocast('cuda', enabled=config.get("mixed_precision", False)):
                # forward pass
                output = model(volume)
                if isinstance(output, tuple):
                    output, output_kd = output
                else:
                    output_kd = output

                if config['eeg_model']["use_age"] == "estimate":
                    output_age = output[:, -1]
                    output = output[:, :-1]

                ####################################### Loss Function ####################################################

                if config["criterion"] == "cross-entropy":
                    loss = mixup_criterion(F.cross_entropy, output, y1, y2, lam)
                elif config["criterion"] == "multi-bce":
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))
                    loss = mixup_criterion(
                        F.binary_cross_entropy_with_logits,
                        output,
                        y1_oh.float(),
                        y2_oh.float(),
                        lam,
                    )
                elif config["criterion"] == "svm":
                    loss = mixup_criterion(F.multi_margin_loss, output, y1, y2, lam)
                elif config["criterion"] == 'focal':
                    y1_oh = F.one_hot(y1, num_classes=output.size(dim=1))
                    y2_oh = F.one_hot(y2, num_classes=output.size(dim=1))

                    loss = mixup_criterion(ops.focal_loss.sigmoid_focal_loss, output, y1_oh.float(), y2_oh.float(), lam,
                                           reduction='mean')
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy', 'multi-bce', 'svm']")

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = output.argmax(dim=-1)
            correct1 = pred.squeeze().eq(y1).sum().item()
            correct2 = pred.squeeze().eq(y2).sum().item()
            correct += lam * correct1 + (1.0 - lam) * correct2
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc
