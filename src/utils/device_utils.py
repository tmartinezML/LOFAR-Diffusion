import pandas as pd
import os
import subprocess
from io import StringIO
import torch
from torch.nn import DataParallel


def physical_gpu_df():
    # Load df with information about GPU memory usage from nvidia-smi
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(
        lambda x: x.rstrip(' [MiB]')
    )
    return gpu_df


def visible_gpus_by_space(renumber=True):
    gpu_df = physical_gpu_df()
    # Sort by free memory
    gpu_df.sort_values(by='memory.free', inplace=True, ascending=False)
    # Reduce to available GPUs by CUDA_VISIBLE_DEVICES (if set)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        vis = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        gpu_df = gpu_df.loc[vis]
        if renumber:
            gpu_df['renumbering'] = [vis.index(i) for i in gpu_df.index]
            gpu_df.set_index('renumbering', inplace=True)

    return list(gpu_df.index)


def distribute_model(model, n_devices=1):
    device_ids = visible_gpus_by_space()[:n_devices]
    if n_devices == 1:
        model.to(torch.device('cuda', device_ids[0]))

    else:
        model.to(torch.device('cuda', device_ids[0]))
        model = DataParallel(
            model,
            device_ids=device_ids
        )
    return model, device_ids


def set_visible_devices(n_gpu):
    max_dev = len(physical_gpu_df())
    assert 0 <= n_gpu <= max_dev, f"n_gpu must be between 0 and {max_dev}"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(i) for i in visible_gpus_by_space()[:n_gpu]]
    )
