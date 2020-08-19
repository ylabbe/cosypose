import yaml
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

from dask_jobqueue import SLURMCluster
from distributed import Client, LocalCluster, as_completed
from .record_chunk import record_chunk

from cosypose.config import CONDA_BASE_DIR, CONDA_ENV, PROJECT_DIR, DASK_LOGS_DIR
from cosypose.config import SLURM_GPU_QUEUE, SLURM_QOS, DASK_NETWORK_INTERFACE

import dask
dask.config.set({'distributed.scheduler.allowed-failures': 1000})


def record_dataset_dask(client, ds_dir,
                        scene_cls, scene_kwargs,
                        n_chunks, n_frames_per_chunk,
                        start_seed=0, resume=False):

    seeds = set(range(start_seed, start_seed + n_chunks))
    if resume:
        done_seeds = (ds_dir / 'seeds_recorded.txt').read_text().strip().split('\n')
        seeds = set(seeds) - set(map(int, done_seeds))
        all_keys = (ds_dir / 'keys_recorded.txt').read_text().strip().split('\n')
    else:
        all_keys = []
    seeds = tuple(seeds)

    future_kwargs = []
    for seed in seeds:
        kwargs = dict(ds_dir=ds_dir, seed=seed,
                      n_frames=n_frames_per_chunk,
                      scene_cls=scene_cls,
                      scene_kwargs=scene_kwargs)
        future_kwargs.append(kwargs)

    futures = []
    for kwargs in future_kwargs:
        futures.append(client.submit(record_chunk, **kwargs))

    iterator = as_completed(futures)
    unit = 'frame'
    unit_scale = n_frames_per_chunk
    n_futures = len(future_kwargs)
    tqdm_iterator = tqdm(iterator, total=n_futures, unit_scale=unit_scale, unit=unit, ncols=80)

    seeds_file = open(ds_dir / 'seeds_recorded.txt', 'a')
    keys_file = open(ds_dir / 'keys_recorded.txt', 'a')

    for future in tqdm_iterator:
        keys, seed = future.result()
        all_keys += keys
        seeds_file.write(f'{seed}\n')
        seeds_file.flush()
        keys_file.write('\n'.join(keys) + '\n')
        keys_file.flush()
        client.cancel(future)

    seeds_file.close()
    keys_file.close()
    return all_keys


def record_dataset(args):
    if args.resume and not args.overwrite:
        resume_args = yaml.load((Path(args.resume) / 'config.yaml').read_text())
        vars(args).update({k: v for k, v in vars(resume_args).items() if 'resume' not in k})

    args.ds_dir = Path(args.ds_dir)
    if args.ds_dir.is_dir():
        if args.resume:
            assert (args.ds_dir / 'seeds_recorded.txt').exists()
        elif args.overwrite:
            shutil.rmtree(args.ds_dir)
        else:
            raise ValueError('There is already a dataset with this name')
    args.ds_dir.mkdir(exist_ok=True)

    (args.ds_dir / 'config.yaml').write_text(yaml.dump(args))

    log_dir = DASK_LOGS_DIR.as_posix()
    if args.distributed:
        env_extra = [
            'module purge',
            f'source {CONDA_BASE_DIR}/bin/activate',
            f'conda activate {CONDA_ENV}',
            f'cd {PROJECT_DIR}',
            f'eval $(python -m job_runner.assign_gpu)',
            'export OMP_NUM_THREADS=1',
            'export MKL_NUM_THREADS=1',
        ]
        n_processes = args.n_processes_per_gpu
        log_path = (DASK_LOGS_DIR / 'all_logs.out').as_posix()

        cluster = SLURMCluster(cores=n_processes,
                               memory='160 GB',
                               queue=f'{SLURM_GPU_QUEUE}',
                               walltime='10:00:00',
                               processes=n_processes,
                               local_directory=log_dir,
                               log_directory=log_dir,
                               nthreads=1,
                               memory_monitor_interval='1000000000000000s',
                               env_extra=env_extra,
                               job_extra=[
                                   f'--qos={SLURM_QOS}',
                                   '--hint=nomultithread',
                                   '--gres=gpu:1',
                                   f'--output={log_path}',
                                   f'--error={log_path}'
                               ],
                               interface=DASK_NETWORK_INTERFACE)
        cluster.adapt(minimum_jobs=args.n_workers, maximum_jobs=args.n_workers)
    else:
        cluster = LocalCluster(local_directory=log_dir, processes=True, n_workers=4)

    client = Client(cluster)

    all_keys = record_dataset_dask(client=client, ds_dir=args.ds_dir,
                                   scene_kwargs=args.scene_kwargs,
                                   scene_cls=args.scene_cls,
                                   start_seed=0,
                                   n_chunks=int(args.n_chunks),
                                   n_frames_per_chunk=int(args.n_frames_per_chunk),
                                   resume=args.resume)

    n_train = int(args.train_ratio * len(all_keys))
    train_keys, val_keys = all_keys[:n_train], all_keys[n_train:]
    Path(args.ds_dir / 'keys.pkl').write_bytes(pickle.dumps(all_keys))
    Path(args.ds_dir / 'train_keys.pkl').write_bytes(pickle.dumps(train_keys))
    Path(args.ds_dir / 'val_keys.pkl').write_bytes(pickle.dumps(val_keys))

    client.close()
    del cluster
