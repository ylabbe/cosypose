import numpy as np
import pybullet as pb

from cosypose.datasets.datasets_cfg import make_urdf_dataset
from cosypose.lib3d import Transform

from cosypose.simulator.base_scene import BaseScene
from cosypose.simulator.caching import BodyCache
from cosypose.simulator.camera import Camera


class BulletSceneRenderer(BaseScene):
    def __init__(self,
                 urdf_ds='ycbv',
                 preload_cache=False,
                 background_color=(0, 0, 0),
                 gpu_renderer=True,
                 gui=False):

        self.urdf_ds = make_urdf_dataset(urdf_ds)
        self.connect(gpu_renderer=gpu_renderer, gui=gui)
        self.body_cache = BodyCache(self.urdf_ds, self.client_id)
        if preload_cache:
            self.body_cache.get_bodies_by_ids(np.arange(len(self.urdf_ds)))
        self.background_color = background_color

    def setup_scene(self, obj_infos):
        labels = [obj['name'] for obj in obj_infos]
        bodies = self.body_cache.get_bodies_by_labels(labels)
        for (obj_info, body) in zip(obj_infos, bodies):
            TWO = Transform(obj_info['TWO'])
            body.pose = TWO
            color = obj_info.get('color', None)
            if color is not None:
                pb.changeVisualShape(body.body_id, -1, physicsClientId=0, rgbaColor=color)
        return bodies

    def render_images(self, cam_infos, render_depth=False):
        cam_obs = []
        for cam_info in cam_infos:
            K = cam_info['K']
            TWC = Transform(cam_info['TWC'])
            resolution = cam_info['resolution']
            cam = Camera(resolution=resolution, client_id=self.client_id)
            cam.set_intrinsic_K(K)
            cam.set_extrinsic_T(TWC)
            cam_obs_ = cam.get_state()
            if self.background_color is not None:
                im = cam_obs_['rgb']
                mask = cam_obs_['mask']
                im[np.logical_or(mask < 0, mask == 255)] = self.background_color
                if render_depth:
                    depth = cam_obs_['depth']
                    near, far = cam_obs_['near'], cam_obs_['far']
                    z_n = 2 * depth - 1
                    z_e = 2 * near * far / (far + near - z_n * (far - near))
                    z_e[np.logical_or(mask < 0, mask == 255)] = 0.
                    cam_obs_['depth'] = z_e
            cam_obs.append(cam_obs_)
        return cam_obs

    def render_scene(self, obj_infos, cam_infos, render_depth=False):
        self.setup_scene(obj_infos)
        return self.render_images(cam_infos, render_depth=render_depth)
