import numpy as np
import logging
import pandas as pd
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse

from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.datasets.bop import remap_bop_targets

from cosypose.evaluation.runner_utils import format_results

from cosypose.training.detector_models_cfg import create_model_detector, check_update_config
from cosypose.integrated.detector import Detector
from cosypose.evaluation.pred_runner.detections import DetectionRunner
from cosypose.scripts.run_cosypose_eval import load_pix2pose_results, load_posecnn_results

from cosypose.evaluation.meters.detection_meters import DetectionMeter
from cosypose.evaluation.eval_runner.detection_eval import DetectionEvaluation

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR
from cosypose.utils.logging import get_logger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)


def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config(cfg)
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


def get_meters(scene_ds):
    visib_gt_min = -1
    n_top = -1  # Given by targets

    if 'bop19' in scene_ds.name:
        targets_filename = 'test_targets_bop19.json'
        targets_path = scene_ds.ds_dir / targets_filename
        targets = pd.read_json(targets_path)
        targets = remap_bop_targets(targets)
    else:
        targets = None

    base_kwargs = dict(
        errors_bsz=64,
        # BOP-Like parameters
        n_top=n_top,
        visib_gt_min=visib_gt_min,
        targets=targets,
    )

    meters = {
        'ntop=BOP': DetectionMeter(**base_kwargs, consider_all_predictions=False),
        'ntop=ALL': DetectionMeter(**base_kwargs, consider_all_predictions=True),
    }
    return meters


def run_detection_eval(args, detector=None):
    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    scene_ds = make_scene_dataset(args.ds_name, n_frames=args.n_frames)

    pred_kwargs = dict()
    pred_runner = DetectionRunner(scene_ds, batch_size=args.pred_bsz,
                                  cache_data=len(pred_kwargs) > 1,
                                  n_workers=args.n_workers)

    if not args.skip_model_predictions:
        if detector is not None:
            model = detector
        else:
            model = load_detector(args.detector_run_id)

        pred_kwargs.update({
            'model': dict(
                detector=model,
                gt_detections=False
            )
        })

    all_predictions = dict()

    if args.external_predictions:
        if 'ycbv' in args.ds_name:
            all_predictions['posecnn'] = load_posecnn_results().cpu()
        elif 'tless' in args.ds_name:
            all_predictions['retinanet/pix2pose'] = load_pix2pose_results(all_detections=True).cpu()
        else:
            pass

    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        logger.info(f"Prediction: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    logger.info("Done with predictions")
    torch.distributed.barrier()

    # Evaluation.
    meters = get_meters(scene_ds)
    logger.info(f"Meters: {meters}")
    eval_runner = DetectionEvaluation(scene_ds, meters, batch_size=args.eval_bsz,
                                      cache_data=len(all_predictions) > 1,
                                      n_workers=args.n_workers,
                                      sampler=pred_runner.sampler)

    eval_metrics, eval_dfs = dict(), dict()
    if not args.skip_evaluation:
        for preds_k, preds in all_predictions.items():
            do_eval = True
            if do_eval:
                logger.info(f"Evaluation of predictions: {preds_k}")
                if len(preds) == 0:
                    preds = eval_runner.make_empty_predictions()
                eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            else:
                logger.info(f"Skipped: {preds_k}")

    for k, v in all_predictions.items():
        all_predictions[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()

    results = None
    if get_rank() == 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Finished evaluation on {args.ds_name}')
        results = format_results(all_predictions, eval_metrics, eval_dfs)
        torch.save(results, save_dir / 'results.pth.tar')
        (save_dir / 'summary.txt').write_text(results.get('summary_txt', ''))
        (save_dir / 'config.yaml').write_text(yaml.dump(args))
        logger.info(f'Saved predictions+metrics in {save_dir}')

    logger.info("Done with evaluation")
    torch.distributed.barrier()
    return results


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--skip_predictions', action='store_true')
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--id', default=-1, type=int)
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--models', default='', type=str)
    args = parser.parse_args()

    init_distributed_mode()

    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.n_workers = 8
    cfg.pred_bsz = 8
    cfg.eval_bsz = 8
    cfg.n_frames = None
    cfg.skip_evaluation = False
    cfg.skip_model_predictions = args.skip_predictions
    cfg.external_predictions = True
    cfg.detector = None
    if args.debug:
        cfg.n_frames = 10

    if args.config == 'bop':
        # ds_names = ['ycbv.bop19', 'tless.bop19']
        ds_names = ['itodd.val', 'hb.val']
    else:
        raise ValueError

    detector_run_ids = {
        'ycbv.bop19': 'ycbv--377940',
        'hb.val': 'detector-bop-hb--497808',
        'itodd.val': 'detector-bop-itodd--509908',
    }

    if args.id < 0:
        n_rand = np.random.randint(1e6)
        args.id = n_rand
    save_dir = RESULTS_DIR / f'{args.config}-{args.models}-{args.comment}-{args.id}'
    logger.info(f'Save dir: {save_dir}')

    for ds_name in ds_names:
        this_cfg = deepcopy(cfg)
        this_cfg.ds_name = ds_name
        this_cfg.save_dir = save_dir / f'dataset={ds_name}'
        logger.info(f'DATASET: {ds_name}')

        if ds_name in detector_run_ids:
            this_cfg.detector_run_id = detector_run_ids[ds_name]
        else:
            this_cfg.skip_model_predictions = True
            logger.info(f'No model provided for dataset: {ds_name}.')
        run_detection_eval(this_cfg)
        logger.info('')


if __name__ == '__main__':
    main()
