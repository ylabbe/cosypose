import numpy as np
import pybullet as pb
import transforms3d

from cosypose.lib3d import Transform
from cosypose.lib3d.rotations import euler2quat


def proj_from_K(K, h, w, near, far):
    # NOTE: inspired from http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    K = K.copy()
    s = K[0, 1]
    alpha = K[0, 0]
    beta = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    y0 = h - y0
    A = near+far
    B = near*far
    persp = np.array([[alpha, s, -x0, 0],
                      [0, beta, -y0, 0],
                      [0, 0, A, B],
                      [0, 0, -1, 0]])
    left, right, bottom, top = 0, w, 0, h
    tx = - (right+left)/(right-left)
    ty = - (top+bottom)/(top-bottom)
    tz = - (far+near) / (far-near)
    NDC = np.array([[2 / (right - left), 0, 0, tx],
                    [0, 2 / (top-bottom), 0, ty],
                    [0, 0, - 2 / (far - near), tz],
                    [0, 0, 0, 1]])
    proj = NDC @ persp
    return proj.T


def K_from_fov(fov, resolution):
    h, w = min(resolution), max(resolution)
    f = h / (2 * np.tan(fov * 0.5 * np.pi / 180))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    return K


class Camera:
    def __init__(self, resolution=(320, 240), near=0.01, far=10, client_id=0):
        assert client_id >= 0, 'Please provide a client id (0 by default)'
        h, w = min(resolution), max(resolution)

        self._client_id = client_id
        self._near = near
        self._far = far
        self._shape = (h, w)
        self._rgba = np.zeros(self._shape + (4,), dtype=np.uint8)
        self._mask = np.zeros(self._shape, dtype=np.uint8)
        self._depth = np.zeros(self._shape, dtype=np.float32)
        self._render_options = dict()
        self._render_flags = 0

        self.mask_link_index(True)
        self.casts_shadow(True)

        # Transform between standard camera coordinate (z forward) and OPENGL camera coordinate system
        wxyz = transforms3d.euler.euler2quat(np.pi / 2, 0, 0, axes='rxyz')
        xyzw = [*wxyz[1:], wxyz[0]]
        self.TCCGL = Transform(xyzw, (0, 0, 0))

        # Set some default parameters
        self.set_extrinsic_bullet(target=(0, 0, 0), distance=1.6, yaw=90, pitch=-35, roll=0)
        self.set_intrinsic_fov(90)

    def set_extrinsic_bullet(self, target, distance, yaw, pitch, roll):
        """
        Angles in *degrees*.
        """
        up = 'z'
        self._view_params = dict(yaw=yaw, pitch=pitch, roll=roll,
                                 target=target, distance=distance)
        self._view_mat = pb.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw, pitch, roll, 'xyz'.index(up))

    def set_extrinsic_T(self, TWC):
        TWC = Transform(TWC)
        TWCGL = TWC * self.TCCGL
        xyzw = TWCGL.quaternion.coeffs()
        wxyz = [xyzw[-1], *xyzw[:-1]]
        pitch, roll, yaw = transforms3d.euler.quat2euler(wxyz, axes='sxyz')
        yaw = yaw * 180 / np.pi
        pitch = pitch * 180 / np.pi
        roll = roll * 180 / np.pi
        yaw = (yaw % 360 + 360) % 360
        distance = 0.0001
        self.set_extrinsic_bullet(target=TWCGL.translation, distance=distance,
                                  pitch=pitch, roll=roll, yaw=yaw)

    def set_extrinsic_spherical(self, target=(0, 0, 0), rho=0.6, theta=np.pi/4, phi=0, roll=0):
        """
        Angles in *radians*.
        https://fr.wikipedia.org/wiki/Coordonn%C3%A9es_sph%C3%A9riques#/media/Fichier:Spherical_Coordinates_(Colatitude,_Longitude)_(b).svg
        """
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        t = np.array([x, y, z])
        R = transforms3d.euler.euler2mat(np.pi, theta, phi, axes='sxyz')
        R = R @ transforms3d.euler.euler2mat(0, 0, -np.pi/2 + roll, axes='sxyz')
        t += np.array(target)
        TWC = Transform(R, t)
        self.set_extrinsic_T(TWC)

    def set_intrinsic_K(self, K):
        h, w = self._shape
        proj_mat = proj_from_K(K, near=self._near, far=self._far, h=h, w=w).flatten()
        assert np.allclose(proj_mat[11], -1)
        self._proj_mat = proj_mat
        self._K = K
        self._proj_params = None

    def set_intrinsic_fov(self, fov):
        h, w = self._shape
        self._proj_params = dict(fov=fov)
        self._proj_mat = pb.computeProjectionMatrixFOV(
            fov=fov, aspect=w / h, nearVal=self._near, farVal=self._far)
        self._K = None

    def set_intrinsic_f(self, *args):
        if len(args) == 2:
            fx, fy = args
            raise NotImplementedError
        else:
            assert len(args) == 1
            fy = args[0]
        h, w = self._shape
        fov_y = np.arctan(h * 0.5 / fy) * 180 / np.pi
        fov = fov_y * 2
        self.set_intrinsic_fov(fov)

    def get_state(self):
        obs = dict()
        # Get images
        rgba, mask, depth = self._shot()
        rgb = rgba[..., :3]
        obs.update(rgb=rgb, mask=mask, depth=depth)

        # Get intrinsic, extrinsic parameters with standard conventions.
        h, w = self._shape
        if self._K is not None:
            K = self._K
        else:
            K = K_from_fov(self._proj_params['fov'])

        trans = self._view_params['target']
        orn = euler2quat([self._view_params[k]*np.pi/180 for k in ('pitch', 'roll', 'yaw')], axes='sxyz')
        TWCGL = Transform(orn, trans)
        TWC = TWCGL * self.TCCGL.inverse()
        obs.update(TWC=TWC.toHomogeneousMatrix(), K=K, resolution=(self._shape[1], self._shape[0]),
                   proj_mat=self._proj_mat, near=self._near, far=self._far)
        return obs

    def _shot(self):
        """ Computes a RGB image, a depth buffer and a segmentation mask buffer
        with body unique ids of visible objects for each pixel.
        """
        h, w = self._shape
        renderer = pb.ER_BULLET_HARDWARE_OPENGL

        w, h, rgba, depth, mask = pb.getCameraImage(
            width=w,
            height=h,
            projectionMatrix=self._proj_mat,
            viewMatrix=self._view_mat,
            renderer=renderer,
            flags=self._render_flags,
            **self._render_options,
            physicsClientId=self._client_id)

        rgba = np.asarray(rgba, dtype=np.uint8).reshape((h, w, 4))
        depth = np.asarray(depth, dtype=np.float32).reshape((h, w))
        mask = np.asarray(mask, dtype=np.uint8).reshape((h, w))
        return rgba, mask, depth

    def _project(self, fov, near, far):
        """ Apply camera projection matrix.
            Args:
             fov (float): Field of view.
             near float): Near plane distance.
             far (float): Far plane distance.
        """
        self.near = near
        self.far = far
        h, w = self._shape
        self._proj_mat = pb.computeProjectionMatrixFOV(
            fov=fov, aspect=w / h, nearVal=near, farVal=far)

    def mask_link_index(self, flag):
        """ If is enabled, the mask combines the object unique id and link index
        as follows: value = objectUniqueId + ((linkIndex+1)<<24).
        """
        if flag:
            self._render_flags |= pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        else:
            self._render_flags &= ~pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

    def casts_shadow(self, flag):
        """ 1 for shadows, 0 for no shadows. """
        self._render_options['shadow'] = 1 if flag else 0
