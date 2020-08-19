import yaml
import argparse
import numpy as np
import time
import torch
import simplejson as json
from tqdm import tqdm
import functools
from torchnet.meter import AverageValueMeter
from collections import defaultdict
import torch.distributed as dist

from cosypose.config import EXP_DIR

from torch.utils.data import DataLoader, ConcatDataset
from cosypose.utils.multiepoch_dataloader import MultiEpochDataLoader
from torchvision.models.utils import load_state_dict_from_url

from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.datasets.detection_dataset import DetectionDataset
from cosypose.datasets.samplers import PartialSampler

from torchvision.models.detection.mask_rcnn import model_urls

from .maskrcnn_forward_loss import h_maskrcnn
from .detector_models_cfg import create_model_detector, check_update_config


from cosypose.utils.logging import get_logger
from cosypose.utils.distributed import get_world_size, get_rank, sync_model, init_distributed_mode, reduce_dict
from torch.backends import cudnn

# Evaluation
from cosypose.scripts.run_detection_eval import run_detection_eval
from cosypose.integrated.detector import Detector

cudnn.benchmark = True
logger = get_logger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


def make_eval_configs(args, model_training, epoch):
    model = model_training.module
    model.config = args
    model.cfg = args
    detector = Detector(model)

    configs = []
    for ds_name in args.test_ds_names:
        cfg = argparse.ArgumentParser('').parse_args([])
        cfg.ds_name = ds_name
        cfg.save_dir = args.save_dir / f'dataset={ds_name}/epoch={epoch}'
        cfg.n_workers = args.n_dataloader_workers
        cfg.pred_bsz = 16
        cfg.eval_bsz = 16
        cfg.n_frames = None
        cfg.skip_evaluation = False
        cfg.skip_model_predictions = False
        cfg.external_predictions = True
        cfg.n_frames = args.n_test_frames
        configs.append(cfg)
    return configs, detector


def run_eval(args, model_training, epoch):
    errors = dict()
    configs, detector = make_eval_configs(args, model_training, epoch)
    for cfg in configs:
        results = run_detection_eval(cfg, detector=detector)
        if dist.get_rank() == 0:
            errors[cfg.ds_name] = results['summary']
    return errors


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


def train_detector(args):
    torch.set_num_threads(1)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / 'config.yaml').read_text())
        keep_fields = set(['resume_run_id', 'epoch_size', ])
        vars(args).update({k: v for k, v in vars(resume_args).items() if k not in keep_fields})

    args = check_update_config(args)
    args.save_dir = EXP_DIR / args.run_id

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
        all_labels = set()
        for (ds_name, n_repeat) in dataset_names:
            assert 'test' not in ds_name
            ds = make_scene_dataset(ds_name)
            logger.info(f'Loaded {ds_name} with {len(ds)} images.')
            all_labels = all_labels.union(set(ds.all_labels))
            for _ in range(n_repeat):
                datasets.append(ds)
        return ConcatDataset(datasets), all_labels

    scene_ds_train, train_labels = make_datasets(args.train_ds_names)
    scene_ds_val, _ = make_datasets(args.val_ds_names)
    label_to_category_id = dict()
    label_to_category_id['background'] = 0
    for n, label in enumerate(sorted(list(train_labels)), 1):
        label_to_category_id[label] = n
    logger.info(f'Training with {len(label_to_category_id)} categories: {label_to_category_id}')
    args.label_to_category_id = label_to_category_id

    ds_kwargs = dict(
        resize=args.input_resize,
        rgb_augmentation=args.rgb_augmentation,
        background_augmentation=args.background_augmentation,
        gray_augmentation=args.gray_augmentation,
        label_to_category_id=label_to_category_id,
    )
    ds_train = DetectionDataset(scene_ds_train, **ds_kwargs)
    ds_val = DetectionDataset(scene_ds_val, **ds_kwargs)

    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(ds_train, sampler=train_sampler, batch_size=args.batch_size,
                               num_workers=args.n_dataloader_workers,
                               collate_fn=collate_fn,
                               drop_last=False, pin_memory=True)
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * args.epoch_size))
    ds_iter_val = DataLoader(ds_val, sampler=val_sampler, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers,
                             collate_fn=collate_fn,
                             drop_last=False, pin_memory=True)
    ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    model = create_model_detector(cfg=args,
                                  n_classes=len(args.label_to_category_id)).cuda()

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
    elif args.pretrain_coco:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        keep = lambda k: 'box_predictor' not in k and 'mask_predictor' not in k
        state_dict = {k: v for k, v in state_dict.items() if keep(k)}
        model.load_state_dict(state_dict, strict=False)
        logger.info('Using model pre-trained on coco. Removed predictor heads.')
    else:
        logger.info('Training MaskRCNN from scratch.')

    # Synchronize models across processes.
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer {args.optimizer}')

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
        meters_train = defaultdict(AverageValueMeter)
        meters_val = defaultdict(AverageValueMeter)
        meters_time = defaultdict(AverageValueMeter)

        h = functools.partial(h_maskrcnn, model=model, cfg=args)

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
                    model.parameters(), max_norm=np.inf, norm_type=2)
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
            model.train()
            for sample in tqdm(ds_iter_val, ncols=80):
                loss = h(data=sample, meters=meters_val)
                meters_val['loss_total'].add(loss.item())

        train_epoch()
        if epoch % args.val_epoch_interval == 0:
            validation()

        test_dict = None
        if epoch % args.test_epoch_interval == 0:
            model.eval()
            test_dict = run_eval(args, model, epoch)

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
