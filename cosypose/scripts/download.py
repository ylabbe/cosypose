import argparse
import zipfile
import wget
import logging
import subprocess
from pathlib import Path
from cosypose.config import PROJECT_DIR, LOCAL_DATA_DIR, BOP_DS_DIR
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

RCLONE_CFG_PATH = (PROJECT_DIR / 'rclone.conf')
RCLONE_ROOT = 'cosypose:'
DOWNLOAD_DIR = LOCAL_DATA_DIR / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)

BOP_SRC = 'http://ptak.felk.cvut.cz/6DB/public/bop_datasets/'
BOP_DATASETS = {
    'ycbv': {
        'splits': ['train_real', 'train_synt', 'test_all']
    },

    'tless': {
        'splits': ['test_primesense_all', 'train_primesense'],
    },

    'hb': {
        'splits': ['test_primesense_all', 'val_primesense'],
    },

    'icbin': {
        'splits': ['test_all'],
    },

    'itodd': {
        'splits': ['val', 'test_all'],
    },

    'lm': {
        'splits': ['test_all'],
    },

    'lmo': {
        'splits': ['test_all'],
        'has_pbr': False,
    },

    'tudl': {
        'splits': ['test_all', 'train_real']
    },
}

BOP_DS_NAMES = list(BOP_DATASETS.keys())


def main():
    parser = argparse.ArgumentParser('CosyPose download utility')
    parser.add_argument('--bop_dataset', default='', type=str, choices=BOP_DS_NAMES)
    parser.add_argument('--bop_src', default='bop', type=str, choices=['bop', 'gdrive'])
    parser.add_argument('--bop_extra_files', default='', type=str, choices=['ycbv', 'tless'])
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--urdf_models', default='', type=str, choices=['ycbv', 'tless.cad'])
    parser.add_argument('--ycbv_compat_models', action='store_true')
    parser.add_argument('--texture_dataset', action='store_true')
    parser.add_argument('--result_id', default='', type=str)
    parser.add_argument('--bop_result_id', default='', type=str)
    parser.add_argument('--synt_dataset', default='', type=str)
    parser.add_argument('--detections', default='', type=str)
    parser.add_argument('--example_scenario', action='store_true')
    parser.add_argument('--pbr_training_images', action='store_true')
    parser.add_argument('--all_bop20_results', action='store_true')
    parser.add_argument('--all_bop20_models', action='store_true')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.bop_dataset:
        if args.bop_src == 'bop':
            download_bop_original(args.bop_dataset, args.pbr_training_images and BOP_DATASETS[args.bop_dataset].get('has_pbr', True))
        elif args.bop_src == 'gdrive':
            download_bop_gdrive(args.bop_dataset)

    if args.bop_extra_files:
        if args.bop_extra_files == 'tless':
            # https://github.com/kirumang/Pix2Pose#download-pre-trained-weights
            gdrive_download(f'bop_datasets/tless/all_target_tless.json', BOP_DS_DIR / 'tless')
        elif args.bop_extra_files == 'ycbv':
            # Friendly names used with YCB-Video
            gdrive_download(f'bop_datasets/ycbv/ycbv_friendly_names.txt', BOP_DS_DIR / 'ycbv')
            # Offsets between YCB-Video and BOP (extracted from BOP readme)
            gdrive_download(f'bop_datasets/ycbv/offsets.txt', BOP_DS_DIR / 'ycbv')
            # Evaluation models for YCB-Video (used by other works)
            gdrive_download(f'bop_datasets/ycbv/models_original', BOP_DS_DIR / 'ycbv')
            # Keyframe definition
            gdrive_download(f'bop_datasets/ycbv/keyframe.txt', BOP_DS_DIR / 'ycbv')

    if args.urdf_models:
        gdrive_download(f'urdfs/{args.urdf_models}', LOCAL_DATA_DIR / 'urdfs')

    if args.ycbv_compat_models:
        gdrive_download(f'bop_datasets/ycbv/models_bop-compat', BOP_DS_DIR / 'ycbv')
        gdrive_download(f'bop_datasets/ycbv/models_bop-compat_eval', BOP_DS_DIR / 'ycbv')

    if args.model:
        gdrive_download(f'experiments/{args.model}', LOCAL_DATA_DIR / 'experiments')

    if args.detections:
        gdrive_download(f'saved_detections/{args.detections}.pkl', LOCAL_DATA_DIR / 'saved_detections')

    if args.result_id:
        gdrive_download(f'results/{args.result_id}', LOCAL_DATA_DIR / 'results')

    if args.bop_result_id:
        csv_name = args.bop_result_id + '.csv'
        gdrive_download(f'bop_predictions/{csv_name}', LOCAL_DATA_DIR / 'bop_predictions')
        gdrive_download(f'bop_eval_outputs/{args.bop_result_id}', LOCAL_DATA_DIR / 'bop_predictions')

    if args.texture_dataset:
        gdrive_download('zip_files/textures.zip', DOWNLOAD_DIR)
        logger.info('Extracting textures ...')
        zipfile.ZipFile(DOWNLOAD_DIR / 'textures.zip').extractall(LOCAL_DATA_DIR / 'texture_datasets')

    if args.synt_dataset:
        zip_name = f'{args.synt_dataset}.zip'
        gdrive_download(f'zip_files/{zip_name}', DOWNLOAD_DIR)
        logger.info('Extracting textures ...')
        zipfile.ZipFile(DOWNLOAD_DIR / zip_name).extractall(LOCAL_DATA_DIR / 'synt_datasets')

    if args.example_scenario:
        gdrive_download(f'custom_scenarios/example/candidates.csv', LOCAL_DATA_DIR / 'custom_scenarios/example')
        gdrive_download(f'custom_scenarios/example/scene_camera.json', LOCAL_DATA_DIR / 'custom_scenarios/example')

    if args.all_bop20_models:
        from cosypose.bop_config import (PBR_DETECTORS, PBR_COARSE, PBR_REFINER,
                                         SYNT_REAL_DETECTORS, SYNT_REAL_COARSE, SYNT_REAL_REFINER)
        for model_dict in (PBR_DETECTORS, PBR_COARSE, PBR_REFINER,
                           SYNT_REAL_DETECTORS, SYNT_REAL_COARSE, SYNT_REAL_REFINER):
            for model in model_dict.values():
                gdrive_download(f'experiments/{model}', LOCAL_DATA_DIR / 'experiments')

    if args.all_bop20_results:
        from cosypose.bop_config import (PBR_INFERENCE_ID, SYNT_REAL_INFERENCE_ID, SYNT_REAL_ICP_INFERENCE_ID,
                                         SYNT_REAL_4VIEWS_INFERENCE_ID, SYNT_REAL_8VIEWS_INFERENCE_ID)
        for result_id in (PBR_INFERENCE_ID, SYNT_REAL_INFERENCE_ID, SYNT_REAL_ICP_INFERENCE_ID,
                          SYNT_REAL_4VIEWS_INFERENCE_ID, SYNT_REAL_8VIEWS_INFERENCE_ID):
            gdrive_download(f'results/{result_id}', LOCAL_DATA_DIR / 'results')


def run_rclone(cmd, args, flags):
    rclone_cmd = ['rclone', cmd] + args + flags + ['--config', str(RCLONE_CFG_PATH)]
    logger.debug(' '.join(rclone_cmd))
    subprocess.run(rclone_cmd)


def gdrive_download(gdrive_path, local_path):
    gdrive_path = Path(gdrive_path)
    if gdrive_path.name != local_path.name:
        local_path = local_path / gdrive_path.name
    rclone_path = RCLONE_ROOT+str(gdrive_path)
    local_path = str(local_path)
    logger.info(f'Copying {rclone_path} to {local_path}')
    run_rclone('copyto', [rclone_path, local_path], flags=['-P'])


def download_bop_original(ds_name, download_pbr):
    filename = f'{ds_name}_base.zip'
    wget_download_and_extract(BOP_SRC + filename, BOP_DS_DIR)

    suffixes = ['models'] + BOP_DATASETS[ds_name]['splits']
    if download_pbr:
        suffixes += ['train_pbr']
    for suffix in suffixes:
        wget_download_and_extract(BOP_SRC + f'{ds_name}_{suffix}.zip', BOP_DS_DIR / ds_name)


def download_bop_gdrive(ds_name):
    gdrive_download(f'bop_datasets/{ds_name}', BOP_DS_DIR / ds_name)


def wget_download_and_extract(url, out):
    tmp_path = DOWNLOAD_DIR / url.split('/')[-1]
    if tmp_path.exists():
        logger.info(f'{url} already downloaded: {tmp_path}...')
    else:
        logger.info(f'Download {url} at {tmp_path}...')
        wget.download(url, out=tmp_path.as_posix())
    logger.info(f'Extracting {tmp_path} at {out}.')
    zipfile.ZipFile(tmp_path).extractall(out)


if __name__ == '__main__':
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    main()
