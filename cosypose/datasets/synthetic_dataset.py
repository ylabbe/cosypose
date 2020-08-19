import torch
import pandas as pd
import numpy as np
import pickle as pkl
import yaml
import cv2
from io import BytesIO
from .utils import make_detections_from_segmentation
from .datasets_cfg import make_urdf_dataset
from pathlib import Path
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class SyntheticSceneDataset:
    def __init__(self, ds_dir, train=True):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists()

        keys_path = ds_dir / (('train' if train else 'val') + '_keys.pkl')
        keys = pkl.loads(keys_path.read_bytes())
        self.cfg = yaml.load((ds_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        self.object_set = self.cfg.scene_kwargs['urdf_ds']
        self.keys = keys

        urdf_ds_name = self.cfg.scene_kwargs['urdf_ds']
        urdf_ds = make_urdf_dataset(urdf_ds_name)
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]
        self.frame_index = pd.DataFrame(dict(scene_id=np.arange(len(keys)), view_id=np.arange(len(keys))))

    def __len__(self):
        return len(self.frame_index)

    @staticmethod
    def _deserialize_im_cv2(im_buf, rgb=True):
        stream = BytesIO(im_buf)
        stream.seek(0)
        file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        if rgb:
            im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            im = im[..., [2, 1, 0]]
        else:
            im = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        return torch.as_tensor(im)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pkl_path = (self.ds_dir / 'dumps' / key).with_suffix('.pkl')
        dic = pkl.loads(pkl_path.read_bytes())

        cam = dic['camera']
        rgb = self._deserialize_im_cv2(cam['rgb'])
        mask = self._deserialize_im_cv2(cam['mask'], rgb=False)
        cam = {k: v for k, v in cam.items() if k not in {'rgb', 'mask'}}
        objects = dic['objects']
        dets_gt = make_detections_from_segmentation(torch.as_tensor(mask).unsqueeze(0))[0]
        mask_uniqs = set(np.unique(mask[mask > 0]))
        for obj in objects:
            if obj['id_in_segm'] in mask_uniqs:
                obj['bbox'] = dets_gt[obj['id_in_segm']].numpy()
        state = dict(camera=cam, objects=objects, frame_info=self.frame_index.iloc[idx].to_dict())
        return rgb, mask, state
