import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sklearn
from medcam import medcam

# __all__ = []


"""IL edited"""
def compute_reconstruction(model, sample_batched, preprocess, config, target_from_last=1):
    model.eval()
    preprocess(sample_batched)

    x = sample_batched["signal"]
    age = sample_batched["age"]

    module = model.module if config.get("ddp", False) else model
    output, _, _ = module.mask_and_reconstruct(x, age, mask_ratio=0.5)
    output = module.unpatchify(output)

    return output



def export_signal(model, sample_batched, preprocess, config):
    output = compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=0)

    return output


def plot_and_save_eeg_signals(eeg_tensor, sample_idx=0, save_path=None):
    """
    20채널 EEG 신호를 시각화하고 저장하는 함수
    :param eeg_tensor: (Batch, Channels, Seq_Length) 형태의 텐서
    :param sample_idx: 시각화할 샘플 인덱스
    :param save_path: 저장할 파일 경로 (PNG 형식)
    """
    assert sample_idx < eeg_tensor.shape[0], "Invalid sample index"
    os.makedirs(save_path, exist_ok=True)

    eeg_sample = eeg_tensor[sample_idx].cpu().detach().numpy()  # (20, 2048)

    fig, axes = plt.subplots(nrows=20, ncols=1, figsize=(15, 15), sharex=True)
    fig.suptitle(f"EEG Sample {sample_idx}", fontsize=16)

    time = np.arange(eeg_sample.shape[1])

    for i, ax in enumerate(axes):
        ax.plot(time, eeg_sample[i], label=f'Channel {i + 1}', color='b', linewidth=0.8)
        ax.set_ylabel(f"Ch {i + 1}", fontsize=10, rotation=0, labelpad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if i == 19:
            ax.set_xlabel("Time (samples)")
        else:
            ax.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if save_path:
        plt.savefig(save_path+"/eeg_plot_"+save_path.split('/')[-1], dpi=300, bbox_inches='tight')
        print(f"✅ EEG plot saved at: {save_path}")

    plt.show()

def vis_signal(model, sample_batched, preprocess, config):
    output = compute_reconstruction(model, sample_batched, preprocess, config, target_from_last=0)
    plot_and_save_eeg_signals(output, save_path="/home/night/Mycode/EEG_night/vis/recon")
    plot_and_save_eeg_signals(sample_batched['signal'],save_path="/home/night/Mycode/EEG_night/vis/origin")
    return output
""""""""""""""

@torch.no_grad()
def compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=1):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess[0](sample_batched)
    preprocess[1](sample_batched)

    # apply model on whole batch directly on device
    signal = sample_batched["signal"]
    volume = sample_batched["volume"]
    age = sample_batched["age"]

    module = model.module if config.get("ddp", False) else model
    output = module.compute_feature_embedding(signal, volume, age, target_from_last=target_from_last)

    # DeiT model
    if isinstance(output, tuple):
        output = (output[0] + output[1]) / 2.0

    return output

@torch.no_grad()
def unimodal_compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=1):
    # evaluation mode
    model.eval()

    # preprocessing (this includes to-device operation)
    preprocess(sample_batched)

    # apply model on whole batch directly on device
    volume = sample_batched["volume"]
    age = sample_batched["age"]

    module = model.module if config.get("ddp", False) else model
    output = module.compute_feature_embedding(volume, target_from_last=target_from_last)

    # DeiT model
    if isinstance(output, tuple):
        output = (output[0] + output[1]) / 2.0

    return output


@torch.no_grad()
def estimate_logit(model, sample_batched, preprocess, config):
    output = compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=0)      ##
    return output


@torch.no_grad()
def logit_to_prob(logit, config):
    # map depending on the model's loss function
    if config["criterion"] == "cross-entropy":
        score = F.softmax(logit, dim=1)
    elif config["criterion"] == "multi-bce":
        score = torch.sigmoid(logit)
    elif config["criterion"] == "svm":
        score = logit
    elif config["criterion"] == "focal":
        score = logit
    else:
        raise ValueError(f"logit_to_prob(): cannot parse config['criterion']={config['criterion']}.")
    return score




@torch.no_grad()
def estimate_class_score(model, sample_batched, preprocess, config):
    output = compute_feature_embedding(model, sample_batched, preprocess, config, target_from_last=0) ###
    output = logit_to_prob(output, config)
    return output


def calculate_confusion_matrix(pred, target, num_classes):
    N = target.shape[0]
    C = num_classes
    confusion = np.zeros((C, C), dtype=np.int32)

    for i in range(N):
        r = target[i]
        c = pred[i]
        confusion[r, c] += 1
    return confusion


def calculate_confusion_matrix2(pred, target, num_classes, set_size):
    N = set_size
    repeat = target.shape[0] // N
    confusion = []
    for r in range(repeat):
        cm = calculate_confusion_matrix(
            pred[r * N : (r + 1) * N],
            target[r * N : (r + 1) * N],
            num_classes=num_classes,
        )
        confusion.append(cm)
    return np.array(confusion)


def calculate_class_wise_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]

    accuracy = np.zeros((n_classes,))
    sensitivity = np.zeros((n_classes,))
    specificity = np.zeros((n_classes,))
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fn = confusion_matrix[c].sum() - tp
        fp = confusion_matrix[:, c].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp

        accuracy[c] = (tp + tn) / (tp + fn + fp + tn)
        sensitivity[c] = tp / (tp + fn)
        specificity[c] = tn / (fp + tn)
        precision[c] = tp / (tp + fp)
        recall[c] = sensitivity[c]
    f1_score = 2 * precision * recall / (precision + recall)

    class_wise_metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
    }  # 'Recall': recall is same with sensitivity
    return class_wise_metrics


@torch.no_grad()
def check_accuracy(model, loader, preprocess, config, repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_class_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_extended(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    preds = []

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_class_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = estimate_class_score(model, sample_batched, preprocess, config)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # for other metrics
            preds += pred

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    # Added 250715
    preds = torch.tensor(preds)
    Precision = sklearn.metrics.precision_score(target, preds, average='macro')
    Recall = sklearn.metrics.recall_score(target, preds, average='macro')
    F1 = sklearn.metrics.f1_score(target, preds, average='macro')
    # AUROC = sklearn.metrics.roc_auc_score(target_prob, pred_prob, average='macro', multi_class='ovr')
    # AUPRC = sklearn.metrics.average_precision_score(target_prob, pred_prob, average='macro')

    # return accuracy, score, target, confusion_matrix, throughput
    return accuracy, score, target, confusion_matrix, throughput, Precision, Recall, F1


def check_signal_IL(model, loader, preprocess, config, repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = vis_signal(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            start_event.record()
            s = vis_signal(model, sample_batched, preprocess, config)
            ############################################################
            ####   여기까지 했다 ######################
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput


@torch.no_grad()
def check_accuracy_extended_debug(model, loader, preprocess, config, repeat=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for error table
    error_table = {data["serial"]: {"GT": data["class_label"].item(), "Pred": [0] * C} for data in loader.dataset}

    # for crop timing
    crop_timing = dict()

    # for ROC curve
    score = None
    target = None

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            s = estimate_class_score(model, sample_batched, preprocess, config)
            y = sample_batched["class_label"]

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # error table
            for n in range(pred.shape[0]):
                serial = sample_batched["serial"][n]
                error_table[serial]["Pred"][pred[n].item()] += 1

            # crop timing
            for n in range(pred.shape[0]):
                ct = sample_batched["crop_timing"][n]

                if ct not in crop_timing.keys():
                    crop_timing[ct] = {}

                if pred[n] == y[n]:
                    crop_timing[ct]["correct"] = crop_timing[ct].get("correct", 0) + 1
                else:
                    crop_timing[ct]["incorrect"] = crop_timing[ct].get("incorrect", 0) + 1

    # error table update
    error_table_serial = []
    error_table_pred = []
    error_table_gt = []

    for serial in sorted(error_table.keys()):
        error_table_serial.append(serial)
        error_table_pred.append(error_table[serial]["Pred"])
        error_table_gt.append(error_table[serial]["GT"])

    error_table = {
        "Serial": error_table_serial,
        "Pred": error_table_pred,
        "GT": error_table_gt,
    }

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    return accuracy, score, target, confusion_matrix, error_table, crop_timing


@torch.no_grad()
def check_accuracy_multicrop(model, loader, preprocess, config, aggregation="prob", repeat=1):
    # for accuracy
    correct, total = (0, 0)

    for k in range(repeat):
        for sample_batched in loader:
            # estimate
            if aggregation == "logit":
                s = estimate_logit(model, sample_batched, preprocess, config)
            elif aggregation in ["prob", "probability"]:
                s = estimate_class_score(model, sample_batched, preprocess, config)
            else:
                raise ValueError(
                    f"check_accuracy_multicrop(aggregation): aggregation option must "
                    f"be one among of ['logit', 'prob', 'probability']."
                )
            y = sample_batched["class_label"]
            tcm = config["test_crop_multiple"]

            # multi-crop averaging
            if s.size(0) % tcm != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={tcm}."
                )

            real_minibatch = s.size(0) // tcm
            s_ = torch.zeros((real_minibatch, s.size(1)))
            y_ = torch.zeros((real_minibatch,), dtype=torch.int32)

            for m in range(real_minibatch):
                s_[m] = s[tcm * m : tcm * (m + 1)].mean(dim=0, keepdims=True)
                y_[m] = y[tcm * m]

            s = s_
            y = y_

            # calculate accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


@torch.no_grad()
def check_accuracy_multicrop_extended(model, loader, preprocess, config, aggregation="prob", repeat=1, dummy=1):
    # for confusion matrix
    C = config["out_dims"]
    confusion_matrix = np.zeros((C, C), dtype=np.int32)

    # for ROC curve
    score = None
    target = None

    # for throughput calculation
    total = 0
    total_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warm-up using dummy round
    for k in range(dummy):
        for sample_batched in loader:
            _ = estimate_class_score(model, sample_batched, preprocess, config)

    for k in range(repeat):
        for sample_batched in loader:
            real_minibatch = sample_batched["signal"].size(0) // config["test_crop_multiple"]
            s_merge = torch.zeros((real_minibatch, config["out_dims"]))
            y_merge = torch.zeros((real_minibatch,), dtype=torch.int32)

            # estimate
            start_event.record()
            if aggregation == "logit":
                s = estimate_logit(model, sample_batched, preprocess, config)
            elif aggregation in ["prob", "probability"]:
                s = estimate_class_score(model, sample_batched, preprocess, config)
            else:
                raise ValueError(
                    f"check_accuracy_multicrop(aggregation): aggregation option must "
                    f"be one among of ['logit', 'prob', 'probability']."
                )
            y = sample_batched["class_label"]
            tcm = config["test_crop_multiple"]

            # multi-crop averaging
            if s.size(0) % tcm != 0:
                raise ValueError(
                    f"check_accuracy_multicrop(): Real minibatch size={y.size(0)} is not multiple of "
                    f"config['test_crop_multiple']={tcm}."
                )

            for m in range(real_minibatch):
                s_merge[m] = s[tcm * m : tcm * (m + 1)].mean(dim=0, keepdims=True)
                y_merge[m] = y[tcm * m]

            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event) / 1000

            if aggregation == "logit":
                # s_merge = logit_to_prob(s_merge, config)
                s_merge = F.softmax(s_merge, dim=1)

            s = s_merge
            y = y_merge

            # classification score for drawing ROC curve
            if score is None:
                score = s.detach().cpu().numpy()
                target = y.detach().cpu().numpy()
            else:
                score = np.concatenate((score, s.detach().cpu().numpy()), axis=0)
                target = np.concatenate((target, y.detach().cpu().numpy()), axis=0)

            # confusion matrix
            pred = s.argmax(dim=-1)
            confusion_matrix += calculate_confusion_matrix(pred, y, num_classes=config["out_dims"])

            # total samples
            total += pred.shape[0]

    accuracy = confusion_matrix.trace() / confusion_matrix.sum() * 100.0
    throughput = total / total_time

    return accuracy, score, target, confusion_matrix, throughput


def checkm3dcam(model, data_loader, output_dir="./cam"):
    model = medcam.inject(model, output_dir=output_dir, save_maps=True)
    model.eval()
    for batch in data_loader:
        output = model(batch)
