import yaml
import numpy as np
import time
import torch
import simplejson as json
from tqdm import tqdm
import functools
from pathlib import Path
from torchnet.meter import AverageValueMeter
from collections import defaultdict
import torch.distributed as dist

from cosypose.config import EXP_DIR

from torch.utils.data import DataLoader, ConcatDataset
from cosypose.utils.multiepoch_dataloader import MultiEpochDataLoader

from cosypose.datasets.datasets_cfg import make_object_dataset, make_scene_dataset
from cosypose.datasets.pose_dataset import PoseDataset
from cosypose.datasets.samplers import PartialSampler, ListSampler

# Evaluation
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.evaluation.pred_runner.multiview_predictions import MultiviewPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation
from cosypose.evaluation.runner_utils import run_pred_eval
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper
from cosypose.scripts.run_cosypose_eval import (
    load_pix2pose_results, load_posecnn_results, get_pose_meters)

from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.lib3d.rigid_mesh_database import MeshDataBase

from .pose_forward_loss import h_pose
from .pose_models_cfg import create_model_pose, check_update_config


from cosypose.utils.logging import get_logger
from cosypose.utils.distributed import get_world_size, get_rank, sync_model, init_distributed_mode, reduce_dict
from torch.backends import cudnn

cudnn.benchmark = True
logger = get_logger(__name__)


def log(config, model,
        log_dict, test_dict, epoch):
    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)
    log_dict.update(epoch=epoch)
    if not (save_dir / 'config.yaml').exists():
        (save_dir / 'config.yaml').write_text(yaml.dump(config))

    def save_checkpoint(model):
        ckpt_name = 'checkpoint'
        ckpt_name += '.pth.tar'
        path = save_dir / ckpt_name
        torch.save({'state_dict': model.module.state_dict(),
                    'epoch': epoch}, path)

    save_checkpoint(model)
    with open(save_dir / 'log.txt', 'a') as f:
        f.write(json.dumps(log_dict, ignore_nan=True) + '\n')

    if test_dict is not None:
        for ds_name, ds_errors in test_dict.items():
            ds_errors['epoch'] = epoch
            with open(save_dir / f'errors_{ds_name}.txt', 'a') as f:
                f.write(json.dumps(test_dict[ds_name], ignore_nan=True) + '\n')

    logger.info(config.run_id)
    logger.info(log_dict)
    logger.info(test_dict)


def make_eval_bundle(args, model_training):
    eval_bundle = dict()
    model_training.cfg = args

    def load_model(run_id):
        if run_id is None:
            return None
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config(cfg)
        model = create_model_pose(cfg, renderer=model_training.renderer,
                                  mesh_db=model_training.mesh_db).cuda().eval()
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')['state_dict']
        model.load_state_dict(ckpt)
        model.eval()
        model.cfg = cfg
        return model

    if args.train_refiner:
        refiner_model = model_training
        coarse_model = load_model(args.coarse_run_id_for_test)
    elif args.train_coarse:
        coarse_model = model_training
        refiner_model = load_model(args.refiner_run_id_for_test)
    else:
        raise ValueError

    predictor = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                          refiner_model=refiner_model)

    base_pred_kwargs = dict(
        pose_predictor=predictor,
        mv_predictor=None,
        skip_mv=True,
    )
    for ds_name in args.test_ds_names:
        assert ds_name in {'ycbv.test.keyframes', 'tless.primesense.test'}
        scene_ds = make_scene_dataset(ds_name, n_frames=args.n_test_frames)
        logger.info(f'TEST: Loaded {ds_name} with {len(scene_ds)} images.')
        scene_ds_pred = MultiViewWrapper(scene_ds, n_views=1)

        # Predictions
        pred_runner = MultiviewPredictionRunner(scene_ds_pred, batch_size=1,
                                                n_workers=args.n_dataloader_workers, cache_data=False)
        detections = None
        pred_kwargs = dict()

        if 'tless' in ds_name:
            detections = load_pix2pose_results(all_detections=False,
                                               remove_incorrect_poses=False).cpu()
            coarse_detections = load_pix2pose_results(all_detections=False,
                                                      remove_incorrect_poses=True).cpu()
            det_k = 'pix2pose_detections'
            coarse_k = 'pix2pose_coarse'

        elif 'ycbv' in ds_name:
            detections = load_posecnn_results().cpu()
            coarse_detections = detections
            det_k = 'posecnn_detections'
            coarse_k = 'posecnn_coarse'

        else:
            raise ValueError(ds_name)

        if refiner_model is not None:
            pred_kwargs.update({
                coarse_k: dict(
                    detections=coarse_detections,
                    use_detections_TCO=True,
                    n_coarse_iterations=0,
                    n_refiner_iterations=1,
                    **base_pred_kwargs,
                )
            })

        if coarse_model is not None:
            pred_kwargs.update({
                det_k: dict(
                    detections=detections,
                    use_detections_TCO=False,
                    n_coarse_iterations=coarse_model.cfg.n_iterations,
                    n_refiner_iterations=1 if refiner_model is not None else 0,
                    **base_pred_kwargs,
                )
            })

        # Evaluation
        meters = get_pose_meters(scene_ds)
        meters = {k.split('_')[0]: v for k, v in meters.items()}
        mv_group_ids = list(iter(pred_runner.sampler))
        scene_ds_ids = np.concatenate(scene_ds_pred.frame_index.loc[mv_group_ids, 'scene_ds_ids'].values)
        sampler = ListSampler(scene_ds_ids)
        eval_runner = PoseEvaluation(scene_ds, meters, batch_size=1, cache_data=True,
                                     n_workers=args.n_dataloader_workers, sampler=sampler)

        save_dir = Path(args.save_dir) / 'eval' / ds_name
        save_dir.mkdir(exist_ok=True, parents=True)
        eval_bundle[ds_name] = (pred_runner, pred_kwargs, eval_runner, save_dir)
    return eval_bundle


def run_eval(eval_bundle, epoch):
    errors = dict()
    for ds_name, bundle in eval_bundle.items():
        pred_runner, pred_kwargs, eval_runner, save_dir = bundle
        results = run_pred_eval(pred_runner, pred_kwargs, eval_runner)
        if dist.get_rank() == 0:
            torch.save(results, save_dir / f'epoch={epoch}.pth.tar')
            errors[ds_name] = results['summary']
    return errors


def train_pose(args):
    torch.set_num_threads(1)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / 'config.yaml').read_text())
        keep_fields = set(['resume_run_id', 'epoch_size', ])
        vars(args).update({k: v for k, v in vars(resume_args).items() if k not in keep_fields})

    args.train_refiner = args.TCO_input_generator == 'gt+noise'
    args.train_coarse = not args.train_refiner
    args.save_dir = EXP_DIR / args.run_id
    args = check_update_config(args)

    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    # Initialize distributed
    device = torch.cuda.current_device()
    init_distributed_mode()
    world_size = get_world_size()
    args.n_gpus = world_size
    args.global_batch_size = world_size * args.batch_size
    logger.info(f'Connection established with {world_size} gpus.')

    # Make train/val datasets
    def make_datasets(dataset_names):
        datasets = []
        for (ds_name, n_repeat) in dataset_names:
            assert 'test' not in ds_name
            ds = make_scene_dataset(ds_name)
            logger.info(f'Loaded {ds_name} with {len(ds)} images.')
            for _ in range(n_repeat):
                datasets.append(ds)
        return ConcatDataset(datasets)

    scene_ds_train = make_datasets(args.train_ds_names)
    scene_ds_val = make_datasets(args.val_ds_names)

    ds_kwargs = dict(
        resize=args.input_resize,
        rgb_augmentation=args.rgb_augmentation,
        background_augmentation=args.background_augmentation,
        min_area=args.min_area,
        gray_augmentation=args.gray_augmentation,
    )
    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = PoseDataset(scene_ds_val, **ds_kwargs)

    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(ds_train, sampler=train_sampler, batch_size=args.batch_size,
                               num_workers=args.n_dataloader_workers, collate_fn=ds_train.collate_fn,
                               drop_last=False, pin_memory=True)
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * args.epoch_size))
    ds_iter_val = DataLoader(ds_val, sampler=val_sampler, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers, collate_fn=ds_val.collate_fn,
                             drop_last=False, pin_memory=True)
    ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    # Make model
    renderer = BulletBatchRenderer(object_set=args.urdf_ds_name, n_workers=args.n_rendering_workers)
    object_ds = make_object_dataset(args.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds).batched(n_sym=args.n_symmetries_batch).cuda().float()

    model = create_model_pose(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()

    eval_bundle = make_eval_bundle(args, model)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        path = resume_dir / 'checkpoint.pth.tar'
        logger.info(f'Loading checkpoing from {path}')
        save = torch.load(path)
        state_dict = save['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = save['epoch'] + 1
    else:
        start_epoch = 0
    end_epoch = args.n_epochs

    if args.run_id_pretrain is not None:
        pretrain_path = EXP_DIR / args.run_id_pretrain / 'checkpoint.pth.tar'
        logger.info(f'Using pretrained model from {pretrain_path}.')
        model.load_state_dict(torch.load(pretrain_path)['state_dict'])

    # Synchronize models across processes.
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup
    if args.n_epochs_warmup == 0:
        lambd = lambda epoch: 1
    else:
        n_batches_warmup = args.n_epochs_warmup * (args.epoch_size // args.batch_size)
        lambd = lambda batch: (batch + 1) / n_batches_warmup
    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambd)
    lr_scheduler_warmup.last_epoch = start_epoch * args.epoch_size // args.batch_size

    # LR schedulers
    # Divide LR by 10 every args.lr_epoch_decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_epoch_decay, gamma=0.1,
    )
    lr_scheduler.last_epoch = start_epoch - 1
    lr_scheduler.step()

    for epoch in range(start_epoch, end_epoch):
        meters_train = defaultdict(lambda: AverageValueMeter())
        meters_val = defaultdict(lambda: AverageValueMeter())
        meters_time = defaultdict(lambda: AverageValueMeter())

        h = functools.partial(h_pose, model=model, cfg=args, n_iterations=args.n_iterations,
                              mesh_db=mesh_db, input_generator=args.TCO_input_generator)

        def train_epoch():
            model.train()
            iterator = tqdm(ds_iter_train, ncols=80)
            t = time.time()
            for n, sample in enumerate(iterator):
                if n > 0:
                    meters_time['data'].add(time.time() - t)

                optimizer.zero_grad()

                t = time.time()
                loss = h(data=sample, meters=meters_train)
                meters_time['forward'].add(time.time() - t)
                iterator.set_postfix(loss=loss.item())
                meters_train['loss_total'].add(loss.item())

                t = time.time()
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)
                meters_train['grad_norm'].add(torch.as_tensor(total_grad_norm).item())

                optimizer.step()
                meters_time['backward'].add(time.time() - t)
                meters_time['memory'].add(torch.cuda.max_memory_allocated() / 1024. ** 2)

                if epoch < args.n_epochs_warmup:
                    lr_scheduler_warmup.step()
                t = time.time()
            if epoch >= args.n_epochs_warmup:
                lr_scheduler.step()

        @torch.no_grad()
        def validation():
            model.eval()
            for sample in tqdm(ds_iter_val, ncols=80):
                loss = h(data=sample, meters=meters_val)
                meters_val['loss_total'].add(loss.item())

        @torch.no_grad()
        def test():
            model.eval()
            return run_eval(eval_bundle, epoch=epoch)

        train_epoch()
        if epoch % args.val_epoch_interval == 0:
            validation()

        test_dict = None
        if epoch % args.test_epoch_interval == 0:
            test_dict = test()

        log_dict = dict()
        log_dict.update({
            'grad_norm': meters_train['grad_norm'].mean,
            'grad_norm_std': meters_train['grad_norm'].std,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time_forward': meters_time['forward'].mean,
            'time_backward': meters_time['backward'].mean,
            'time_data': meters_time['data'].mean,
            'gpu_memory': meters_time['memory'].mean,
            'time': time.time(),
            'n_iterations': (epoch + 1) * len(ds_iter_train),
            'n_datas': (epoch + 1) * args.global_batch_size * len(ds_iter_train),
        })

        for string, meters in zip(('train', 'val'), (meters_train, meters_val)):
            for k in dict(meters).keys():
                log_dict[f'{string}_{k}'] = meters[k].mean

        log_dict = reduce_dict(log_dict)
        if get_rank() == 0:
            log(config=args, model=model, epoch=epoch,
                log_dict=log_dict, test_dict=test_dict)
        dist.barrier()
