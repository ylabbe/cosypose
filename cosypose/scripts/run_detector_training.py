import argparse
import numpy as np
import os
from colorama import Fore, Style

from cosypose.training.train_detector import train_detector
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    cfg = argparse.ArgumentParser('').parse_args([])
    if args.config:
        logger.info(f"{Fore.GREEN}Training with config: {args.config} {Style.RESET_ALL}")

    cfg.resume_run_id = None
    if len(args.resume) > 0:
        cfg.resume_run_id = args.resume
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")

    N_CPUS = int(os.environ.get('N_CPUS', 10))
    N_GPUS = int(os.environ.get('N_PROCS', 1))
    N_WORKERS = min(N_CPUS - 2, 8)
    N_RAND = np.random.randint(1e6)
    cfg.n_gpus = N_GPUS

    run_comment = ''

    # Data
    cfg.train_ds_names = []
    cfg.val_ds_names = cfg.train_ds_names
    cfg.val_epoch_interval = 10
    cfg.test_ds_names = []
    cfg.test_epoch_interval = 30
    cfg.n_test_frames = None

    cfg.input_resize = (480, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'resnet50-fpn'
    cfg.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    # Pretraning
    cfg.run_id_pretrain = None
    cfg.pretrain_coco = True

    # Training
    cfg.batch_size = 2
    cfg.epoch_size = 5000
    cfg.n_epochs = 600
    cfg.lr_epoch_decay = 200
    cfg.n_epochs_warmup = 50
    cfg.n_dataloader_workers = N_WORKERS

    # Optimizer
    cfg.optimizer = 'sgd'
    cfg.lr = (0.02 / 8) * N_GPUS * float(cfg.batch_size / 4)
    cfg.weight_decay = 1e-4
    cfg.momentum = 0.9

    # Method
    cfg.rpn_box_reg_alpha = 1
    cfg.objectness_alpha = 1
    cfg.classifier_alpha = 1
    cfg.mask_alpha = 1
    cfg.box_reg_alpha = 1

    if 'tless' in args.config:
        cfg.input_resize = (540, 720)
    elif 'ycbv' in args.config:
        cfg.input_resize = (480, 640)
    elif 'bop-' in args.config:
        cfg.input_resize = None
    else:
        raise ValueError

    if 'bop-' in args.config:
        from cosypose.bop_config import BOP_CONFIG
        from cosypose.bop_config import PBR_DETECTORS
        bop_name, train_type = args.config.split('-')[1:]
        bop_cfg = BOP_CONFIG[bop_name]
        if train_type == 'pbr':
            cfg.train_ds_names = [(bop_cfg['train_pbr_ds_name'][0], 1)]
        elif train_type == 'synt+real':
            cfg.train_ds_names = bop_cfg['train_synt_real_ds_names']
            cfg.run_id_pretrain = PBR_DETECTORS[bop_name]
        else:
            raise ValueError
        cfg.val_ds_names = cfg.train_ds_names
        cfg.input_resize = bop_cfg['input_resize']
        if len(bop_cfg['test_ds_name']) > 0:
            cfg.test_ds_names = bop_cfg['test_ds_name']

    else:
        raise ValueError(args.config)
    cfg.val_ds_names = cfg.train_ds_names

    if args.no_eval:
        cfg.test_ds_names = []

    cfg.run_id = f'detector-{args.config}-{run_comment}-{N_RAND}'

    if args.debug:
        cfg.n_epochs = 4
        cfg.val_epoch_interval = 1
        cfg.batch_size = 2
        cfg.epoch_size = 10 * cfg.batch_size
        cfg.run_id = 'debug-' + cfg.run_id
        cfg.background_augmentation = False
        cfg.rgb_augmentation = False
        cfg.n_dataloader_workers = 1
        cfg.n_test_frames = 10

    N_GPUS = int(os.environ.get('N_PROCS', 1))
    cfg.epoch_size = cfg.epoch_size // N_GPUS

    train_detector(cfg)
