import os
import subprocess
import time

import jax
import jax.numpy as jnp


def spread_over_devices(x, devices=None, as_sharded_array=True):
    """
    Converts a single-device jnp array to a distributed jnp array.

    From: https://github.com/instadeepai/poppy/blob/main/poppy/utils/utils.py
    """
    devices = devices or jax.local_devices()

    def distribute_fn(x):
        x = x.reshape(len(devices), -1, *(x.shape[1:]))
        x = list(x)
        if as_sharded_array:
            x = jax.device_put_sharded(x, devices)
        return x

    return jax.tree_util.tree_map(distribute_fn, x)


def toggled_pmap(func, debug_pmap=False, axis_name=None):
    def wrapper(*args, **kwargs):
        if debug_pmap:
            return func(*args, **kwargs)
        else:
            return jax.pmap(func, axis_name=axis_name)(*args, **kwargs)

    return wrapper


def flatten_over_devices(x):
    """
    Converts an array of shape [num_devices, dim2, dim3, ...] into an array
    of shape [num_devices * dim2, dim3, ...].
    """
    return jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), x)


def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    return gpus_to_use
