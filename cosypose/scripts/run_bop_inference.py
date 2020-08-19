import logging
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse

from cosypose.bop_config import BOP_CONFIG
from cosypose.bop_config import PBR_COARSE, PBR_REFINER, PBR_DETECTORS
from cosypose.bop_config import SYNT_REAL_COARSE, SYNT_REAL_REFINER, SYNT_REAL_DETECTORS
from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

from cosypose.evaluation.runner_utils import format_results

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.icp_refiner import ICPRefiner
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR, RESULTS_DIR
from cosypose.utils.logging import get_logger


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)


def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)

    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def run_inference(args):
    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    scene_ds = make_scene_dataset(args.ds_name, n_frames=args.n_frames)

    if args.icp:
        scene_ds.load_depth = args.icp

    # if args.debug and 'tless' in args.ds_name:
    #     # Try to debug ICP on T-LESS ??????
    #     view_id = 142
    #     mask = scene_ds.frame_index['view_id'] == view_id
    #     scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)

    #     scene_id = 1
    #     mask = scene_ds.frame_index['scene_id'] == scene_id
    #     scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)

    scene_ds_multi = MultiViewWrapper(scene_ds, n_views=args.n_views)

    if args.n_groups is not None:
        scene_ds_multi.frame_index = scene_ds_multi.frame_index[:args.n_groups].reset_index(drop=True)

    pred_kwargs = dict()
    pred_runner = BopPredictionRunner(scene_ds_multi, batch_size=args.pred_bsz,
                                      cache_data=False, n_workers=args.n_workers)

    detector = load_detector(args.detector_run_id)
    pose_predictor, mesh_db = load_pose_models(
        coarse_run_id=args.coarse_run_id,
        refiner_run_id=args.refiner_run_id,
        n_workers=args.n_workers,
    )

    icp_refiner = None
    if args.icp:
        renderer = pose_predictor.coarse_model.renderer
        icp_refiner = ICPRefiner(mesh_db,
                                 renderer=renderer,
                                 resolution=pose_predictor.coarse_model.cfg.input_resize)

    mv_predictor = None
    if args.n_views > 1:
        mv_predictor = MultiviewScenePredictor(mesh_db)

    pred_kwargs.update({
        'maskrcnn_detections': dict(
            detector=detector,
            pose_predictor=pose_predictor,
            n_coarse_iterations=args.n_coarse_iterations,
            n_refiner_iterations=args.n_refiner_iterations,
            icp_refiner=icp_refiner,
            mv_predictor=mv_predictor,
        )
    })

    all_predictions = dict()
    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        logger.info(f"Prediction: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    logger.info("Done with inference.")
    torch.distributed.barrier()

    for k, v in all_predictions.items():
        all_predictions[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()

    if get_rank() == 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Finished inference on {args.ds_name}')
        results = format_results(all_predictions, dict(), dict())
        torch.save(results, save_dir / 'results.pth.tar')
        (save_dir / 'config.yaml').write_text(yaml.dump(args))
        logger.info(f'Saved predictions in {save_dir}')

    torch.distributed.barrier()
    return


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--id', default=-1, type=int)
    parser.add_argument('--config', default='bop-pbr', type=str)
    parser.add_argument('--nviews', dest='n_views', default=1, type=int)
    parser.add_argument('--icp', action='store_true')
    args = parser.parse_args()

    init_distributed_mode()

    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.n_workers = 8
    cfg.pred_bsz = 1
    cfg.n_frames = None
    cfg.n_groups = None
    cfg.skip_evaluation = False
    cfg.external_predictions = True

    cfg.n_coarse_iterations = 1
    cfg.n_refiner_iterations = 4
    cfg.icp = args.icp
    cfg.debug = args.debug
    cfg.n_views = args.n_views
    if args.debug:
        if args.n_views > 1:
            cfg.n_groups = 1
        else:
            cfg.n_frames = 4
        # cfg.n_workers = 1

    if args.id < 0:
        n_rand = np.random.randint(1e6)
        args.id = n_rand

    if args.icp:
        args.comment = f'icp-{args.comment}'

    if args.n_views > 1:
        args.comment = f'nviews={args.n_views}-{args.comment}'

    save_dir = RESULTS_DIR / f'{args.config}-{args.comment}-{args.id}'
    logger.info(f'Save dir: {save_dir}')

    if args.config == 'bop-pbr':
        MODELS_DETECTORS = PBR_DETECTORS
        MODELS_COARSE = PBR_COARSE
        MODELS_REFINER = PBR_REFINER

    elif args.config == 'bop-synt+real':
        MODELS_DETECTORS = SYNT_REAL_DETECTORS
        MODELS_COARSE = SYNT_REAL_COARSE
        MODELS_REFINER = SYNT_REAL_REFINER

    if args.n_views > 1:
        ds_names = ['hb', 'tless', 'ycbv']
    else:
        ds_names = ['hb', 'icbin', 'itodd', 'lmo', 'tless', 'tudl', 'ycbv']

    for ds_name in ds_names:
        this_cfg = deepcopy(cfg)
        this_cfg.ds_name = BOP_CONFIG[ds_name]['inference_ds_name'][0]
        this_cfg.save_dir = save_dir / f'dataset={ds_name}'

        this_cfg.detector_run_id = MODELS_DETECTORS.get(ds_name)
        this_cfg.coarse_run_id = MODELS_COARSE.get(ds_name)
        this_cfg.refiner_run_id = MODELS_REFINER.get(ds_name)

        if this_cfg.detector_run_id is None \
           or this_cfg.coarse_run_id is None \
           or this_cfg.refiner_run_id is None:
            logger.info(f'Skipped {ds_name}')
            continue

        run_inference(this_cfg)


if __name__ == '__main__':
    main()
