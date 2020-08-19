import numpy as np
import pandas as pd

import cosypose.utils.tensor_collection as tc

from cosypose.lib3d.transform_ops import invert_T
from cosypose.multiview.ransac import multiview_candidate_matching
from cosypose.multiview.bundle_adjustment import make_view_groups, MultiviewRefinement

from cosypose.utils.logging import get_logger
logger = get_logger(__name__)


class MultiviewScenePredictor:
    def __init__(self, mesh_db, n_sym=64, ba_aabb=True, ba_n_points=None):
        self.mesh_db_ransac = mesh_db.batched(n_sym=n_sym, aabb=True).cuda().float()
        self.mesh_db_ba = mesh_db.batched(
            aabb=ba_aabb, resample_n_points=ba_n_points, n_sym=n_sym).cuda().float()

    def reproject_scene(self, objects, cameras):
        TCO_data = []
        for o in range(len(objects)):
            for v in range(len(cameras)):
                obj = objects[[o]]
                cam = cameras[[v]]
                infos = dict(
                    scene_id=cam.infos['scene_id'].values,
                    view_id=cam.infos['view_id'].values,
                    score=obj.infos['score'].values + 1.0,
                    view_group=obj.infos['view_group'].values,
                    label=obj.infos['label'].values,
                    batch_im_id=cam.infos['batch_im_id'].values,
                    obj_id=obj.infos['obj_id'].values,
                    from_ba=[True],
                )
                data_ = tc.PandasTensorCollection(
                    infos=pd.DataFrame(infos),
                    poses=invert_T(cam.TWC) @ obj.TWO,
                )
                TCO_data.append(data_)
        return tc.concatenate(TCO_data)

    def predict_scene_state(
            self, candidates, cameras,
            score_th=0.3, use_known_camera_poses=False,
            ransac_n_iter=2000, ransac_dist_threshold=0.02,
            ba_n_iter=100):

        predictions = dict()
        cand_inputs = candidates

        assert len(np.unique(candidates.infos['scene_id'])) == 1
        scene_id = np.unique(candidates.infos['scene_id']).item()
        group_id = np.unique(candidates.infos['group_id']).item()
        keep = np.where(candidates.infos['score'] >= score_th)[0]
        candidates = candidates[keep]

        predictions['cand_inputs'] = candidates

        logger.debug(f'Num candidates: {len(candidates)}')
        logger.debug(f'Num views: {len(cameras)}')

        matching_outputs = multiview_candidate_matching(
            candidates=candidates, mesh_db=self.mesh_db_ransac,
            n_ransac_iter=ransac_n_iter, dist_threshold=ransac_dist_threshold,
            cameras=cameras if use_known_camera_poses else None
        )

        pairs_TC1C2 = matching_outputs['pairs_TC1C2']
        candidates = matching_outputs['filtered_candidates']

        logger.debug(f'Matched candidates: {len(candidates)}')
        for k, v in matching_outputs.items():
            if 'time' in k:
                logger.debug(f'RANSAC {k}: {v}')

        predictions['cand_matched'] = candidates

        group_infos = make_view_groups(pairs_TC1C2)
        candidates = candidates.merge_df(group_infos, on='view_id').cuda()

        pred_objects, pred_cameras, pred_reproj = [], [], []
        pred_reproj_init = []
        for (view_group, candidate_ids) in candidates.infos.groupby('view_group').groups.items():
            candidates_n = candidates[candidate_ids]
            problem = MultiviewRefinement(candidates=candidates_n,
                                          cameras=cameras,
                                          pairs_TC1C2=pairs_TC1C2,
                                          mesh_db=self.mesh_db_ba)
            ba_outputs = problem.solve(
                n_iterations=ba_n_iter,
                optimize_cameras=not use_known_camera_poses,
            )
            pred_objects_, pred_cameras_ = ba_outputs['objects'], ba_outputs['cameras']
            for x in (pred_objects_, pred_cameras_):
                x.infos['view_group'] = view_group
                x.infos['group_id'] = group_id
                x.infos['scene_id'] = scene_id
            pred_reproj.append(self.reproject_scene(pred_objects_, pred_cameras_))
            pred_objects.append(pred_objects_)
            pred_cameras.append(pred_cameras_)

            pred_objects_init, pred_cameras_init = ba_outputs['objects_init'], ba_outputs['cameras_init']
            for x in (pred_objects_init, pred_cameras_init):
                x.infos['view_group'] = view_group
                x.infos['group_id'] = group_id
                x.infos['scene_id'] = scene_id
            pred_reproj_init.append(self.reproject_scene(pred_objects_init, pred_cameras_init))

            for k, v in ba_outputs.items():
                if 'time' in k:
                    logger.debug(f'BA {k}: {v}')

        predictions['scene/objects'] = tc.concatenate(pred_objects)
        predictions['scene/cameras'] = tc.concatenate(pred_cameras)

        predictions['ba_output'] = tc.concatenate(pred_reproj)
        predictions['ba_input'] = tc.concatenate(pred_reproj_init)

        cand_inputs = tc.PandasTensorCollection(
            infos=cand_inputs.infos,
            poses=cand_inputs.poses,
        )
        predictions['ba_output+all_cand'] = tc.concatenate(
            [predictions['ba_output'], cand_inputs],
        )
        return predictions
