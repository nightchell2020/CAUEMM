from typing import List, Dict

import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# __all__ = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def program_conv_filters(
    sequence_length: int,
    conv_filter_list: List[Dict],
    output_lower_bound: int,
    output_upper_bound: int,
    pad: bool = True,
    stride_to_pool_ratio: float = 1.00,
    trials: int = 5,
    class_name: str = "",
    verbose=False,
):
    # desired
    mid = (output_upper_bound + output_lower_bound) / 2.0
    in_out_ratio = float(sequence_length) / mid

    base_stride = np.power(
        in_out_ratio / np.prod([cf["kernel_size"] for cf in conv_filter_list], dtype=np.float64),
        1.0 / len(conv_filter_list),
    )

    for i in range(len(conv_filter_list)):
        cf = conv_filter_list[i]
        if i == 0 and len(conv_filter_list) > 1:
            total_stride = max(1.0, base_stride * cf["kernel_size"] * 0.7)
            cf["pool"] = max(
                1,
                round(np.sqrt(total_stride / stride_to_pool_ratio) * stride_to_pool_ratio * 0.3),
            )
            cf["stride"] = max(1, round(total_stride / cf["pool"]))
        else:
            total_stride = max(1.0, base_stride * cf["kernel_size"])
            if stride_to_pool_ratio > 1.0:
                cf["pool"] = min(
                    max(
                        1,
                        round(np.sqrt(total_stride / stride_to_pool_ratio) * stride_to_pool_ratio),
                    ),
                    round(total_stride),
                )
                cf["stride"] = max(1, round(total_stride / cf["pool"]))
            else:
                cf["stride"] = min(
                    max(1, round(np.sqrt(total_stride / stride_to_pool_ratio))),
                    round(total_stride),
                )
                cf["pool"] = max(1, round(total_stride / cf["stride"]))

        # cf['r'] = np.sqrt(total_stride / stride_pool_ratio)
        # cf['dilation'] = 1
        conv_filter_list[i] = cf

    success = False
    str_debug = f"\n{'-'*100}\nstarting from sequence length: {sequence_length}\n{'-'*100}\n"
    current_length = sequence_length

    for k in range(trials):
        if success:
            break

        for pivot in reversed(range(len(conv_filter_list))):
            current_length = sequence_length

            for cf in conv_filter_list:
                current_length = current_length // cf.get("pool", 1)
                str_debug += f"{cf} >> {current_length} "

                effective_kernel_size = (cf["kernel_size"] - 1) * cf.get("dilation", 1)
                both_side_pad = 2 * (cf["kernel_size"] // 2) if pad is True else 0
                current_length = (current_length + both_side_pad - effective_kernel_size - 1) // cf["stride"] + 1
                str_debug += f">> {current_length}\n"

            pool = conv_filter_list[pivot]["pool"]
            stride = conv_filter_list[pivot]["stride"]
            if current_length < output_lower_bound:
                if float(pool) / stride < stride_to_pool_ratio:
                    if stride > 1:
                        conv_filter_list[pivot]["stride"] = max(1, stride - 1)
                    else:
                        conv_filter_list[pivot]["pool"] = max(1, pool - 1)
                else:
                    if pool > 1:
                        conv_filter_list[pivot]["pool"] = max(1, pool - 1)
                    else:
                        conv_filter_list[pivot]["stride"] = max(1, stride - 1)
            elif current_length > output_upper_bound:
                if float(pool) / stride < stride_to_pool_ratio:
                    conv_filter_list[pivot]["pool"] = pool + 1
                else:
                    conv_filter_list[pivot]["stride"] = stride + 1
            else:
                str_debug += f">> Success!"
                success = True
                break

            str_debug += f">> Failed.."
            str_debug += f"\n{'-' * 100}\n"

    if verbose:
        print(str_debug)

    if not success:
        header = class_name + ", " if len(class_name) > 0 else ""
        raise RuntimeError(
            f"{header}conv_filter_programming() failed to determine "
            f"the proper convolution filter parameters. "
            f"The following is the recording for debug: {str_debug}"
        )

    output_length = current_length
    return output_length


def visualize_network_tensorboard(model, train_loader, device, nb_fname, name):
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter("runs/" + nb_fname + "_" + name)

    for batch_i, sample_batched in enumerate(train_loader):
        # pull up the batch data
        x = sample_batched["signal"].to(device)
        age = sample_batched["age"].to(device)

        # apply model on whole batch directly on device
        writer.add_graph(model, (x, age))
        break

    writer.close()


def make_pool_or_not(base_pool, pool: int):
    def do_nothing(x):
        return x

    if pool == 1:
        return do_nothing
    elif pool > 1:
        return base_pool(pool)
    else:
        raise ValueError(f"make_pool_or_not(pool) receives an invalid value as input.")

def compute_3d_output_size(input_size, conv_list):
    out = np.array(input_size)

    for cf in conv_list:
        kernel = cf["kernel_size"]
        stride = cf["stride"]
        dilation = cf.get("dilation", 1)
        pool = cf.get("pool", 1)
        pad = cf.get("pad", [1,1,1])

        # Pooling
        out = out // pool

        for i in range(3):
            effective_k = dilation * (kernel[i] - 1)
            out[i] = (out[i] + pad[i] - effective_k - 1) // stride[i] + 1

    return tuple(out)

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    model_size = total_size / (1024 ** 2)
    print(f"📦 Model size: {model_size:.2f} MB")


