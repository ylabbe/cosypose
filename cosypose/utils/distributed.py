import sys
import os
import torch.distributed as dist
import torch
from pathlib import Path


def get_tmp_dir():
    if 'JOB_DIR' in os.environ:
        tmp_dir = Path(os.environ['JOB_DIR']) / 'tmp'
    else:
        tmp_dir = Path('/tmp/cosypose_job')
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def sync_model(model):
    sync_dir = get_tmp_dir() / 'models'
    sync_dir.mkdir(exist_ok=True)
    sync_ckpt = sync_dir / 'sync.checkpoint'
    if get_rank() == 0 and get_world_size() > 1:
        torch.save(model.state_dict(), sync_ckpt)
    dist.barrier()
    if get_rank() > 0:
        model.load_state_dict(torch.load(sync_ckpt))
    dist.barrier()
    return model


def redirect_output():
    if 'JOB_DIR' in os.environ:
        rank = get_rank()
        output_file = Path(os.environ['JOB_DIR']) / f'stdout{rank}.out'
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file, 'w')
    return


def get_rank():
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
    return rank


def get_world_size():
    if not torch.distributed.is_initialized():
        world_size = 1
    else:
        world_size = torch.distributed.get_world_size()
    return world_size


def init_distributed_mode(initfile=None):
    assert torch.cuda.device_count() == 1
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    if initfile is None:
        initfile = get_tmp_dir() / 'initfile'
        if initfile.exists() and world_size == 1:
            initfile.unlink()
    initfile = Path(initfile)
    assert initfile.parent.exists()
    torch.distributed.init_process_group(
        backend='nccl', rank=rank, world_size=world_size,
        init_method=f'file://{initfile.as_posix()}'
    )
    torch.distributed.barrier()


def reduce_dict(input_dict, average=True):
    """
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        reduced_dict = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.tensor(values).float().cuda()
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict
