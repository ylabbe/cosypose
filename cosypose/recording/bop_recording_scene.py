from pathlib import Path
import pinocchio as pin
import numpy as np

from cosypose.config import ASSET_DIR

from cosypose.datasets.datasets_cfg import make_urdf_dataset, make_texture_dataset

from cosypose.simulator import BaseScene, Body, Camera
from cosypose.simulator import BodyCache, TextureCache, apply_random_textures


class SamplerError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class BopRecordingScene(BaseScene):
    def __init__(self,
                 urdf_ds='ycbv',
                 texture_ds='shapenet',
                 domain_randomization=True,
                 textures_on_objects=False,
                 n_objects_interval=(2, 5),
                 objects_xyz_interval=((0.0, -0.5, -0.15), (1.0, 0.5, 0.15)),
                 proba_falling=0.5,
                 resolution=(640, 480),
                 focal_interval=((515, 515), (515, 515)),
                 camera_distance_interval=(0.5, 1.5),
                 border_check=True,
                 gpu_renderer=True,
                 n_textures_cache=50,
                 seed=0):

        # Objects
        self.urdf_ds = make_urdf_dataset(urdf_ds)
        self.n_objects_interval = n_objects_interval
        self.objects_xyz_interval = objects_xyz_interval
        self.n_objects_cache = len(self.urdf_ds)
        assert self.n_objects_cache >= max(n_objects_interval)
        self.away_transform = (0, 0, 1000), (0, 0, 0, 1)
        self.proba_falling = proba_falling

        # Domain randomization
        self.texture_ds = make_texture_dataset(texture_ds)
        self.n_textures_cache = min(n_textures_cache, len(self.texture_ds))
        self.domain_randomization = domain_randomization
        self.textures_on_objects = textures_on_objects

        # Camera
        self.resolution = resolution
        self.focal_interval = np.array(focal_interval)
        self.camera_distance_interval = camera_distance_interval
        self.border_check = border_check
        self.gpu_renderer = gpu_renderer

        # Seeding
        self.np_random = np.random.RandomState(seed)
        pin.seed(seed)
        self.seed = seed

    def load_background(self):
        cage_path = Path(ASSET_DIR / 'cage' / 'cage.urdf').as_posix()
        self.background = Body.load(cage_path, client_id=self.client_id, scale=3.0)

    def load_plane(self):
        plane_path = Path(ASSET_DIR / 'plane' / 'plane.urdf').as_posix()
        self.plane = Body.load(plane_path, client_id=self.client_id, scale=2.0)

    def background_pos_orn_rand(self):
        pos = self.np_random.uniform(np.ones(3) * -1, np.ones(3))
        orn = pin.Quaternion(pin.SE3.Random().rotation).coeffs()
        self.background.pose = pos, orn

    def show_plane(self):
        self.plane.pose = (0, 0, 0), (0, 0, 0, 1)

    def hide_plane(self):
        self.plane.pose = self.away_transform

    def load_body_cache(self):
        assert self._connected
        self.body_cache = BodyCache(self.urdf_ds, self.client_id)

    def load_texture_cache(self):
        assert self._connected
        ds_texture_ids = self.np_random.choice(len(self.texture_ds), size=self.n_textures_cache)
        self.texture_cache = TextureCache(self.texture_ds, self.client_id)
        [self.texture_cache.get_texture(idx) for idx in ds_texture_ids]

    def connect(self, load=True):
        super().connect(gpu_renderer=self.gpu_renderer)

        if load:
            self.load_background()
            self.load_plane()
            self.hide_plane()
            self.load_body_cache()
            self.load_texture_cache()

    def disconnect(self):
        super().disconnect()

    def pick_rand_objects(self):
        n_min, n_max = self.n_objects_interval
        n_objects = self.np_random.choice(np.arange(n_min, n_max + 1))
        ids = self.np_random.choice(len(self.urdf_ds), size=n_objects, replace=False)
        self.bodies = self.body_cache.get_bodies_by_ids(ids)

    def visuals_rand(self):
        bodies = [self.background] + [self.plane]
        if self.textures_on_objects and self.np_random.rand() > 0.9:
            bodies = self.bodies + bodies
        for body in bodies:
            apply_random_textures(body, self.texture_cache.cached_textures,
                                  np_random=self.np_random)

    def objects_pos_orn_rand(self):
        self.hide_plane()
        for body in self.bodies:
            pos = self.np_random.uniform(*self.objects_xyz_interval)
            orn = pin.Quaternion(pin.SE3.Random().rotation).coeffs()
            body.pose = pos, orn

    def objects_pos_orn_rand_falling(self):
        self.show_plane()
        dheight = 0.05
        for n, body in enumerate(self.bodies):
            pos = self.np_random.uniform(*self.objects_xyz_interval)
            pos[2] = dheight * (n + 1)
            orn = pin.Quaternion(pin.SE3.Random().rotation).coeffs()
            body.pose = pos, orn

        ms = self.np_random.randint(5, 10) * 100
        self.run_simulation(float(ms) * 1e-3)

    def sample_camera(self):
        assert self.focal_interval.shape == (2, 2)
        K = np.zeros((3, 3), dtype=np.float)
        fxfy = self.np_random.uniform(*self.focal_interval)
        W, H = max(self.resolution), min(self.resolution)
        K[0, 0] = fxfy[0]
        K[1, 1] = fxfy[1]
        K[0, 2] = W / 2
        K[1, 2] = H / 2
        K[2, 2] = 1.0
        rho = self.np_random.uniform(*self.camera_distance_interval)
        theta = self.np_random.uniform(0, np.pi/2)
        phi = self.np_random.uniform(0, 2 * np.pi)
        roll = self.np_random.uniform(-10, 10) * np.pi / 180
        box_center = np.mean(self.objects_xyz_interval, axis=0)

        cam = Camera(resolution=self.resolution, client_id=self._client_id)
        cam.set_intrinsic_K(K)
        cam.set_extrinsic_spherical(target=box_center, rho=rho, phi=phi, theta=theta, roll=roll)
        return cam

    def camera_rand(self):
        N = 0
        valid = False
        self.cam_obs = None
        while not valid:
            cam = self.sample_camera()
            cam_obs_ = cam.get_state()
            mask = cam_obs_['mask']
            mask[mask == self.background._body_id] = 0
            mask[mask == 255] = 0
            uniqs = np.unique(cam_obs_['mask'])

            valid = len(uniqs) == len(self.bodies) + 1
            if valid and self.border_check:
                for uniq in uniqs[uniqs > 0]:
                    H, W = cam_obs_['mask'].shape
                    ids = np.where(cam_obs_['mask'] == uniq)
                    if ids[0].max() == H-1 or ids[0].min() == 0 or \
                       ids[1].max() == W-1 or ids[1].min() == 0:
                        valid = False
            N += 1
            if N >= 3:
                raise SamplerError('Cannot sample valid camera configuration.')
            self.cam_obs = cam_obs_

    def _full_rand(self,
                   objects=True,
                   objects_pos_orn=True,
                   falling=False,
                   background_pos_orn=True,
                   camera=True,
                   visuals=True):
        if background_pos_orn:
            self.background_pos_orn_rand()
        if objects:
            self.pick_rand_objects()
        if visuals:
            self.visuals_rand()
        if objects_pos_orn:
            if falling:
                self.objects_pos_orn_rand_falling()
            else:
                self.objects_pos_orn_rand()
        if camera:
            self.camera_rand()

    def get_state(self):
        objects = []
        for body in self.bodies:
            state = body.get_state()
            state['id_in_segm'] = body._body_id
            objects.append(state)

        state = dict(
            camera=self.cam_obs,
            objects=objects,
        )
        return state

    def try_rand(self):
        n_iter = 0
        while n_iter < 50:
            try:
                falling = self.np_random.rand() < self.proba_falling
                visuals = self.domain_randomization
                background_pos_orn = self.domain_randomization
                kwargs = dict(
                    objects=True,
                    objects_pos_orn=True,
                    falling=falling,
                    background_pos_orn=background_pos_orn,
                    camera=True,
                    visuals=visuals,
                )
                self._full_rand(**kwargs)
                return
            except SamplerError as e:
                print("Sampling failed: ", e)
                n_iter += 1
        raise SamplerError('Sampling failed')

    def make_new_scene(self):
        self.try_rand()
        obs = self.get_state()
        return obs
