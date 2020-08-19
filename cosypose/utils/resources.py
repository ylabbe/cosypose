import os
import psutil
from shutil import which


def is_egl_available():
    return is_gpu_available and 'EGL_VISIBLE_DEVICES' in os.environ


def is_gpu_available():
    return which('nvidia-smi') is not None


def is_slurm_available():
    return which('sinfo') is not None


def get_total_memory():
    current_process = psutil.Process(os.getpid())
    mem = current_process.memory_info().rss
    for child in current_process.children(recursive=True):
        mem += child.memory_info().rss
    return mem / 1e9
