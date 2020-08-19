import torch
import sys
import pandas as pd
import argparse
from pathlib import Path
import json
import numpy as np
import logging

from cosypose.datasets.bop_object_datasets import BOPObjectDataset
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
import cosypose.utils.tensor_collection as tc
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.multiview import make_cosypose_plots
from cosypose.visualization.multiview import make_scene_renderings, nms3d
from cosypose.utils.logging import get_logger
from cosypose.config import BOP_TOOLKIT_DIR, LOCAL_DATA_DIR

sys.path.append(str(BOP_TOOLKIT_DIR))
from bop_toolkit_lib import inout  # noqa

logger = get_logger(__name__)


def tc_to_csv(predictions, csv_path):
    preds = []
    for n in range(len(predictions)):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = -1.0
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    inout.save_bop_results(csv_path, preds)


def read_csv_candidates(csv_path):
    df = pd.read_csv(csv_path)
    infos = df.loc[:, ['im_id', 'scene_id', 'score', 'obj_id']]
    infos['obj_id'] = infos['obj_id'].apply(lambda x: f'obj_{x:06d}')
    infos = infos.rename(dict(im_id='view_id', obj_id='label'), axis=1)
    R = np.stack(df['R'].apply(lambda x: list(map(float, x.split(' '))))).reshape(-1, 3, 3)
    t = np.stack(df['t'].apply(lambda x: list(map(float, x.split(' '))))).reshape(-1, 3) * 1e-3
    R = torch.tensor(R, dtype=torch.float)
    t = torch.tensor(t, dtype=torch.float)
    TCO = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(R), 1, 1)
    TCO[:, :3, :3] = R
    TCO[:, :3, -1] = t
    candidates = tc.PandasTensorCollection(poses=TCO, infos=infos)
    return candidates


def read_cameras(json_path, view_ids):
    cameras = json.loads(Path(json_path).read_text())
    all_K = []
    for view_id in view_ids:
        cam_info = cameras[str(view_id)]
        K = np.array(cam_info['cam_K']).reshape(3, 3)
        all_K.append(K)
    K = torch.as_tensor(np.stack(all_K))
    cameras = tc.PandasTensorCollection(K=K, infos=pd.DataFrame(dict(view_id=view_ids)))
    return cameras


def save_scene_json(objects, cameras, results_scene_path):
    list_cameras = []
    list_objects = []

    for n in range(len(objects)):
        obj = objects.infos.loc[n, ['score', 'label', 'n_cand']].to_dict()
        obj = {k: np.asarray(v).item() for k, v in obj.items()}
        obj['TWO'] = objects.TWO[n].cpu().numpy().tolist()
        list_objects.append(obj)

    for n in range(len(cameras)):
        cam = cameras.infos.loc[n, ['view_id']].to_dict()
        cam['TWC'] = cameras.TWC[n].cpu().numpy().tolist()
        cam['K'] = cameras.K[n].cpu().numpy().tolist()
        list_cameras.append(cam)

    scene = dict(objects=list_objects, cameras=list_cameras)
    results_scene_path.write_text(json.dumps(scene))
    return


def main():
    parser = argparse.ArgumentParser('CosyPose multi-view reconstruction for a custom scenario')
    parser.add_argument('--scenario', default='', type=str, help='Id of the scenario, matching directory must be in local_data/scenarios')
    parser.add_argument('--sv_score_th', default=0.3, type=int, help="Score to filter single-view predictions")
    parser.add_argument('--n_symmetries_rot', default=64, type=int, help="Number of discretized symmetries to use for continuous symmetries")
    parser.add_argument('--ransac_n_iter', default=2000, type=int,
                        help="Max number of RANSAC iterations per pair of views")
    parser.add_argument('--ransac_dist_threshold', default=0.02, type=float,
                        help="Threshold (in meters) on symmetric distance to consider a tentative match an inlier")
    parser.add_argument('--ba_n_iter', default=10, type=int,
                        help="Maximum number of LM iterations in stage 3")
    parser.add_argument('--nms_th', default=0.04, type=float,
                        help='Threshold (meter) for NMS 3D')
    parser.add_argument('--no_visualization', action='store_true')
    args = parser.parse_args()

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    scenario_dir = LOCAL_DATA_DIR / 'custom_scenarios' / args.scenario

    candidates = read_csv_candidates(scenario_dir / 'candidates.csv').float().cuda()
    candidates.infos['group_id'] = 0
    scene_ids = np.unique(candidates.infos['scene_id'])
    assert len(scene_ids) == 1, 'Please only provide 6D pose estimations that correspond to the same scene.'
    scene_id = scene_ids.item()
    view_ids = np.unique(candidates.infos['view_id'])
    n_views = len(view_ids)
    logger.info(f'Loaded {len(candidates)} candidates in {n_views} views.')

    cameras = read_cameras(scenario_dir / 'scene_camera.json', view_ids).float().cuda()
    cameras.infos['scene_id'] = scene_id
    cameras.infos['batch_im_id'] = np.arange(len(view_ids))
    logger.info(f'Loaded cameras intrinsics.')

    object_ds = BOPObjectDataset(scenario_dir / 'models')
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    logger.info(f'Loaded {len(object_ds)} 3D object models.')

    logger.info('Running stage 2 and 3 of CosyPose...')
    mv_predictor = MultiviewScenePredictor(mesh_db)
    predictions = mv_predictor.predict_scene_state(candidates, cameras,
                                                   score_th=args.sv_score_th,
                                                   use_known_camera_poses=False,
                                                   ransac_n_iter=args.ransac_n_iter,
                                                   ransac_dist_threshold=args.ransac_dist_threshold,
                                                   ba_n_iter=args.ba_n_iter)

    objects = predictions['scene/objects']
    cameras = predictions['scene/cameras']
    reproj = predictions['ba_output']

    for view_group in np.unique(objects.infos['view_group']):
        objects_ = objects[np.where(objects.infos['view_group'] == view_group)[0]]
        cameras_ = cameras[np.where(cameras.infos['view_group'] == view_group)[0]]
        reproj_ = reproj[np.where(reproj.infos['view_group'] == view_group)[0]]
        objects_ = nms3d(objects_, th=args.nms_th, poses_attr='TWO')

        view_group_dir = scenario_dir / 'results' / f'subscene={view_group}'
        view_group_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f'Subscene {view_group} has {len(objects_)} objects and {len(cameras_)} cameras.')

        predicted_scene_path = view_group_dir / 'predicted_scene.json'
        scene_reprojected_path = view_group_dir / 'scene_reprojected.csv'
        save_scene_json(objects_, cameras_, predicted_scene_path)
        tc_to_csv(reproj_, scene_reprojected_path)

        logger.info(f'Wrote predicted scene (objects+cameras): {predicted_scene_path}')
        logger.info(f'Wrote predicted objects with pose expressed in camera frame: {scene_reprojected_path}')

        # if args.no_visualization:
        #     logger.info('Skipping visualization.')
        #     continue

        # if not (scenario_dir / 'urdfs').exists():
        #     logger.info('Skipping visualization (URDFS not provided).')
        #     continue

        # logger.info('Generating visualization ...')


if __name__ == '__main__':
    main()
