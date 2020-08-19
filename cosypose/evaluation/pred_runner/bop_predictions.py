import pandas as pd
import time
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


class BopPredictionRunner:
    def __init__(self, scene_ds, batch_size=1, cache_data=False, n_workers=4):

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        assert batch_size == 1
        sampler = DistributedSceneSampler(scene_ds,
                                          num_replicas=self.world_size,
                                          rank=self.rank)
        self.sampler = sampler
        dataloader = DataLoader(scene_ds, batch_size=batch_size,
                                num_workers=n_workers,
                                sampler=sampler,
                                collate_fn=self.collate_fn)

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader
        self.load_depth = scene_ds.scene_ds.load_depth

    def collate_fn(self, batch):
        cam_infos, K = [], []
        im_infos = []
        depth = []
        batch_im_id = -1
        for n, data in enumerate(batch):
            assert n == 0
            images, masks, obss = data
            for c, obs in enumerate(obss):
                batch_im_id += 1
                frame_info = obs['frame_info']
                im_info = {k: frame_info[k] for k in ('scene_id', 'view_id', 'group_id')}
                im_info.update(batch_im_id=batch_im_id)
                im_infos.append(im_info)
                cam_info = im_info.copy()

                K.append(obs['camera']['K'])
                cam_infos.append(cam_info)
                if self.load_depth:
                    depth.append(torch.tensor(obs['camera']['depth']))

        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
        )
        data = dict(
            cameras=cameras,
            images=images,
            im_infos=im_infos,
        )
        if self.load_depth:
            data['depth'] = torch.stack(depth)
        return data

    def get_predictions(self,
                        detector,
                        pose_predictor,
                        icp_refiner=None,
                        mv_predictor=None,
                        n_coarse_iterations=1,
                        n_refiner_iterations=1,
                        detection_th=0.0):

        predictions = defaultdict(list)
        use_icp = icp_refiner is not None
        for n, data in enumerate(tqdm(self.dataloader)):
            images = data['images'].cuda().float().permute(0, 3, 1, 2) / 255
            cameras = data['cameras'].cuda().float()
            im_infos = data['im_infos']
            depth = None
            if self.load_depth:
                depth = data['depth'].cuda().float()
            logger.debug(f"{'-'*80}")
            logger.debug(f"Predictions on {data['im_infos']}")

            def get_preds():
                torch.cuda.synchronize()
                start = time.time()
                this_batch_detections = detector.get_detections(
                    images=images, one_instance_per_class=False, detection_th=detection_th,
                    output_masks=use_icp, mask_th=0.9
                )
                for key in ('scene_id', 'view_id', 'group_id'):
                    this_batch_detections.infos[key] = this_batch_detections.infos['batch_im_id'].apply(lambda idx: im_infos[idx][key])

                all_preds = dict()
                if len(this_batch_detections) > 0:
                    final_preds, all_preds = pose_predictor.get_predictions(
                        images, cameras.K, detections=this_batch_detections,
                        n_coarse_iterations=n_coarse_iterations,
                        n_refiner_iterations=n_refiner_iterations,
                    )

                    if len(images) > 1:
                        mv_preds = mv_predictor.predict_scene_state(
                            final_preds, cameras,
                        )
                        all_preds['multiview'] = mv_preds['ba_output+all_cand']
                        final_preds = all_preds['multiview']

                    if use_icp:
                        all_preds['icp'] = icp_refiner.refine_poses(final_preds, this_batch_detections.masks, depth, cameras)

                torch.cuda.synchronize()
                duration = time.time() - start
                n_dets = len(this_batch_detections)

                logger.debug(f'Full predictions: {n_dets} detections + pose estimation in {duration:.3f} s')
                logger.debug(f"{'-'*80}")
                return this_batch_detections, all_preds, duration

            # Run once without measuring timing
            if n == 0:
                get_preds()
            this_batch_detections, all_preds, duration = get_preds()
            duration = duration / len(images)  # Divide by number of views in multi-view

            if use_icp:
                this_batch_detections.delete_tensor('masks')  # Saves memory when saving

            # NOTE: time isn't correct for n iterations < max number of iterations
            for k, v in all_preds.items():
                v.infos = v.infos.loc[:, ['scene_id', 'view_id', 'label', 'score']]
                v.infos['time'] = duration
                predictions[k].append(v.cpu())
            predictions['detections'].append(this_batch_detections.cpu())

        predictions = dict(predictions)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions
