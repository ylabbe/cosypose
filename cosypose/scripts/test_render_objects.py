import numpy as np
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from tqdm import tqdm
import torch


if __name__ == '__main__':
    # obj_ds_name = 'hb'
    obj_ds_name = 'itodd'
    renderer = BulletSceneRenderer(obj_ds_name, gpu_renderer=True)
    TCO = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, 0.3],
        [0, 0, 0, 1]
    ]).numpy()

    fx, fy = 300, 300
    cx, cy = 320, 240
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])
    cam = dict(
        resolution=(640, 480),
        K=K,
        TWC=np.eye(4)
    )

    all_images = []
    labels = renderer.urdf_ds.index['label'].tolist()
    for n, obj_label in tqdm(enumerate(np.random.permutation(labels))):
        obj = dict(
            name=obj_label,
            TWO=TCO,
        )
        renders = renderer.render_scene([obj], [cam])[0]['rgb']
        assert renders.sum() > 0, obj_label
