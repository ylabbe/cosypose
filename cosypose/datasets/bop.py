import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cosypose.config import MEMORY, BOP_TOOLKIT_DIR
from cosypose.lib3d import Transform
from cosypose.utils.logging import get_logger

import sys
sys.path.append(str(BOP_TOOLKIT_DIR))
from bop_toolkit_lib import inout  # noqa
sys.path = sys.path[:-1]


logger = get_logger(__name__)


def remap_bop_targets(targets):
    targets = targets.rename(columns={'im_id': 'view_id'})
    targets['label'] = targets['obj_id'].apply(lambda x: f'obj_{x:06d}')
    return targets


@MEMORY.cache
def build_index(ds_dir, save_file, split, save_file_annotations):
    scene_ids, cam_ids, view_ids = [], [], []

    annotations = dict()
    base_dir = ds_dir / split

    for scene_dir in base_dir.iterdir():
        scene_id = scene_dir.name
        annotations_scene = dict()
        for f in ('scene_camera.json', 'scene_gt_info.json', 'scene_gt.json'):
            path = (scene_dir / f)
            if path.exists():
                annotations_scene[f.split('.')[0]] = json.loads(path.read_text())
        annotations[scene_id] = annotations_scene
        # for view_id in annotations_scene['scene_gt_info'].keys():
        for view_id in annotations_scene['scene_camera'].keys():
            cam_id = 'cam'
            scene_ids.append(int(scene_id))
            cam_ids.append(cam_id)
            view_ids.append(int(view_id))

    frame_index = pd.DataFrame({'scene_id': scene_ids, 'cam_id': cam_ids,
                                'view_id': view_ids, 'cam_name': cam_ids})
    frame_index.to_feather(save_file)
    save_file_annotations.write_bytes(pickle.dumps(annotations))
    return


class BOPDataset:
    def __init__(self, ds_dir, split='train', load_depth=False):
        ds_dir = Path(ds_dir)
        self.ds_dir = ds_dir
        assert ds_dir.exists(), 'Dataset does not exists.'

        self.split = split
        self.base_dir = ds_dir / split

        logger.info(f'Building index and loading annotations...')
        save_file_index = self.ds_dir / f'index_{split}.feather'
        save_file_annotations = self.ds_dir / f'annotations_{split}.pkl'
        build_index(
            ds_dir=ds_dir, save_file=save_file_index,
            save_file_annotations=save_file_annotations,
            split=split)
        self.frame_index = pd.read_feather(save_file_index).reset_index(drop=True)
        self.annotations = pickle.loads(save_file_annotations.read_bytes())

        models_infos = json.loads((ds_dir / 'models' / 'models_info.json').read_text())
        self.all_labels = [f'obj_{int(obj_id):06d}' for obj_id in models_infos.keys()]
        self.load_depth = load_depth

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, frame_id):
        row = self.frame_index.iloc[frame_id]
        scene_id, view_id = row.scene_id, row.view_id
        view_id = int(view_id)
        view_id_str = f'{view_id:06d}'
        scene_id_str = f'{int(scene_id):06d}'
        scene_dir = self.base_dir / scene_id_str

        rgb_dir = scene_dir / 'rgb'
        if not rgb_dir.exists():
            rgb_dir = scene_dir / 'gray'
        rgb_path = rgb_dir / f'{view_id_str}.png'
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix('.jpg')
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix('.tif')

        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        rgb = torch.as_tensor(rgb)

        cam_annotation = self.annotations[scene_id_str]['scene_camera'][str(view_id)]
        if 'cam_R_w2c' in cam_annotation:
            RC0 = np.array(cam_annotation['cam_R_w2c']).reshape(3, 3)
            tC0 = np.array(cam_annotation['cam_t_w2c']) * 0.001
            TC0 = Transform(RC0, tC0)
        else:
            TC0 = Transform(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation['cam_K']).reshape(3, 3)
        T0C = TC0.inverse()
        T0C = T0C.toHomogeneousMatrix()
        camera = dict(T0C=T0C, K=K, TWC=T0C, resolution=rgb.shape[:2])

        T0C = TC0.inverse()

        objects = []
        mask = np.zeros((h, w), dtype=np.uint8)
        if 'scene_gt_info' in self.annotations[scene_id_str]:
            annotation = self.annotations[scene_id_str]['scene_gt'][str(view_id)]
            n_objects = len(annotation)
            visib = self.annotations[scene_id_str]['scene_gt_info'][str(view_id)]
            for n in range(n_objects):
                RCO = np.array(annotation[n]['cam_R_m2c']).reshape(3, 3)
                tCO = np.array(annotation[n]['cam_t_m2c']) * 0.001
                TCO = Transform(RCO, tCO)
                T0O = T0C * TCO
                T0O = T0O.toHomogeneousMatrix()
                obj_id = annotation[n]['obj_id']
                name = f'obj_{int(obj_id):06d}'
                bbox_visib = np.array(visib[n]['bbox_visib'])
                x, y, w, h = bbox_visib
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                obj = dict(label=name, name=name, TWO=T0O, T0O=T0O,
                           visib_fract=visib[n]['visib_fract'],
                           id_in_segm=n+1, bbox=[x1, y1, x2, y2])
                objects.append(obj)

            mask_path = scene_dir / 'mask_visib' / f'{view_id_str}_all.png'
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
            else:
                for n in range(n_objects):
                    mask_n = np.array(Image.open(scene_dir / 'mask_visib' / f'{view_id_str}_{n:06d}.png'))
                    mask[mask_n == 255] = n + 1

        mask = torch.as_tensor(mask)

        if self.load_depth:
            depth_path = scene_dir / 'depth' / f'{view_id_str}.png'
            if not depth_path.exists():
                depth_path = depth_path.with_suffix('.tif')
            depth = np.array(inout.load_depth(depth_path))
            camera['depth'] = depth * cam_annotation['depth_scale'] / 1000

        obs = dict(
            objects=objects,
            camera=camera,
            frame_info=row.to_dict(),
        )
        return rgb, mask, obs
