import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

from cosypose.utils.logging import get_logger
from cosypose.datasets.samplers import DistributedSceneSampler
import cosypose.utils.tensor_collection as tc
from cosypose.utils.distributed import get_world_size, get_rank, get_tmp_dir

from torch.utils.data import DataLoader

logger = get_logger(__name__)


class MultiviewPredictionRunner:
    def __init__(self, scene_ds, batch_size=1, cache_data=False, n_workers=4):

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        assert batch_size == 1, 'Multiple view groups not supported for now.'
        sampler = DistributedSceneSampler(scene_ds, num_replicas=self.world_size, rank=self.rank)
        self.sampler = sampler
        dataloader = DataLoader(scene_ds, batch_size=batch_size,
                                num_workers=n_workers,
                                sampler=sampler,
                                collate_fn=self.collate_fn)

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

    def collate_fn(self, batch):
        batch_im_id = -1

        cam_infos, K = [], []
        det_infos, bboxes = [], []
        for n, data in enumerate(batch):
            assert n == 0
            images, masks, obss = data
            for c, obs in enumerate(obss):
                batch_im_id += 1
                frame_info = obs['frame_info']
                im_info = {k: frame_info[k] for k in ('scene_id', 'view_id', 'group_id')}
                im_info.update(batch_im_id=batch_im_id)
                cam_info = im_info.copy()

                K.append(obs['camera']['K'])
                cam_infos.append(cam_info)

                for o, obj in enumerate(obs['objects']):
                    obj_info = dict(
                        label=obj['name'],
                        score=1.0,
                    )
                    obj_info.update(im_info)
                    bboxes.append(obj['bbox'])
                    det_infos.append(obj_info)

        gt_detections = tc.PandasTensorCollection(
            infos=pd.DataFrame(det_infos),
            bboxes=torch.as_tensor(np.stack(bboxes)),
        )
        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
        )
        data = dict(
            images=images,
            cameras=cameras,
            gt_detections=gt_detections,
        )
        return data

    def get_predictions(self, pose_predictor, mv_predictor,
                        detections=None,
                        n_coarse_iterations=1, n_refiner_iterations=1,
                        sv_score_th=0.0, skip_mv=True,
                        use_detections_TCO=False):

        assert detections is not None
        if detections is not None:
            mask = (detections.infos['score'] >= sv_score_th)
            detections = detections[np.where(mask)[0]]
            detections.infos['det_id'] = np.arange(len(detections))
            det_index = detections.infos.set_index(['scene_id', 'view_id']).sort_index()

        predictions = defaultdict(list)
        for data in tqdm(self.dataloader):
            images = data['images'].cuda().float().permute(0, 3, 1, 2) / 255
            cameras = data['cameras'].cuda().float()
            gt_detections = data['gt_detections'].cuda().float()

            scene_id = np.unique(gt_detections.infos['scene_id'])
            view_ids = np.unique(gt_detections.infos['view_id'])
            group_id = np.unique(gt_detections.infos['group_id'])
            n_gt_dets = len(gt_detections)
            logger.debug(f"{'-'*80}")
            logger.debug(f'Scene: {scene_id}')
            logger.debug(f'Views: {view_ids}')
            logger.debug(f'Group: {group_id}')
            logger.debug(f'Image has {n_gt_dets} gt detections. (not used)')

            if detections is not None:
                keep_ids, batch_im_ids = [], []
                for group_name, group in cameras.infos.groupby(['scene_id', 'view_id']):
                    if group_name in det_index.index:
                        other_group = det_index.loc[group_name]
                        keep_ids_ = other_group['det_id']
                        batch_im_id = np.unique(group['batch_im_id']).item()
                        batch_im_ids.append(np.ones(len(keep_ids_)) * batch_im_id)
                        keep_ids.append(keep_ids_)
                if len(keep_ids) > 0:
                    keep_ids = np.concatenate(keep_ids)
                    batch_im_ids = np.concatenate(batch_im_ids)
                detections_ = detections[keep_ids]
                detections_.infos['batch_im_id'] = np.array(batch_im_ids).astype(np.int)
            else:
                raise ValueError('No detections')
            detections_ = detections_.cuda().float()
            detections_.infos['group_id'] = group_id.item()

            sv_preds, mv_preds = dict(), dict()
            if len(detections_) > 0:
                data_TCO_init = detections_ if use_detections_TCO else None
                detections__ = detections_ if not use_detections_TCO else None
                candidates, sv_preds = pose_predictor.get_predictions(
                    images, cameras.K, detections=detections__,
                    n_coarse_iterations=n_coarse_iterations,
                    data_TCO_init=data_TCO_init,
                    n_refiner_iterations=n_refiner_iterations,
                )
                candidates.register_tensor('initial_bboxes', detections_.bboxes)

                if not skip_mv:
                    mv_preds = mv_predictor.predict_scene_state(
                        candidates, cameras,
                    )
            logger.debug(f"{'-'*80}")

            for k, v in sv_preds.items():
                predictions[k].append(v.cpu())

            for k, v in mv_preds.items():
                predictions[k].append(v.cpu())

        predictions = dict(predictions)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions
