import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

from cosypose.datasets.samplers import DistributedSceneSampler
import cosypose.utils.tensor_collection as tc
from cosypose.utils.distributed import get_world_size, get_rank, get_tmp_dir

from torch.utils.data import DataLoader


class DetectionRunner:
    def __init__(self, scene_ds, batch_size=8, cache_data=False, n_workers=4):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        sampler = DistributedSceneSampler(scene_ds, num_replicas=self.world_size, rank=self.rank)
        self.sampler = sampler
        dataloader = DataLoader(scene_ds, batch_size=batch_size,
                                num_workers=n_workers,
                                sampler=sampler, collate_fn=self.collate_fn)

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

    def collate_fn(self, batch):
        batch_im_id = -1
        det_infos, bboxes = [], []
        images = []
        im_infos = []
        for n, data in enumerate(batch):
            rgb, masks, obs = data
            batch_im_id += 1
            frame_info = obs['frame_info']
            im_info = {k: frame_info[k] for k in ('scene_id', 'view_id')}
            im_info.update(batch_im_id=batch_im_id)
            im_infos.append(im_info)
            images.append(rgb)

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
            bboxes=torch.as_tensor(np.stack(bboxes)).float(),
        )
        data = dict(
            images=torch.stack(images),
            gt_detections=gt_detections,
            im_infos=im_infos,
        )
        return data

    def get_predictions(self,
                        detector,
                        gt_detections=False):

        predictions = defaultdict(list)

        for data in tqdm(self.dataloader):
            images = data['images'].cuda().float().permute(0, 3, 1, 2) / 255

            if gt_detections:
                preds = data['gt_detections']
            else:
                preds = detector.get_detections(
                    images=images,
                    one_instance_per_class=False,
                )

            im_infos = data['im_infos']
            for k in ('scene_id', 'view_id'):
                preds.infos[k] = preds.infos['batch_im_id'].apply(lambda idx: im_infos[idx][k])

            predictions['detections'].append(preds)

        for k, v in predictions.items():
            predictions[k] = tc.concatenate(predictions[k])
        return predictions
