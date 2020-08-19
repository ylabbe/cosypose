import os
import torch
from cosypose.utils.distributed import init_distributed_mode, get_world_size, get_tmp_dir, get_rank
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)


if __name__ == '__main__':
    init_distributed_mode()
    proc_id = get_rank()
    n_tasks = get_world_size()
    n_cpus = os.environ.get('N_CPUS', 'not specified')
    logger.info(f'Number of processes (=num GPUs): {n_tasks}')
    logger.info(f'Process ID: {proc_id}')
    logger.info(f'TMP Directory for this job: {get_tmp_dir()}')
    logger.info(f'GPU CUDA ID: {torch.cuda.current_device()}')
    logger.info(f'Max number of CPUs for this process: {n_cpus}')
