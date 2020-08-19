from pathlib import Path

import pybullet as pb
from cosypose.lib3d import Transform, parse_pose_args

from .client import BulletClient


class Body:
    def __init__(self, body_id, scale=1.0, client_id=0):
        self._body_id = body_id
        self._client = BulletClient(client_id)
        self._scale = scale

    @property
    def name(self):
        info = self._client.getBodyInfo(self._body_id)
        return info[-1].decode('utf8')

    @property
    def pose(self):
        return self.pose

    @pose.getter
    def pose(self):
        pos, orn = self._client.getBasePositionAndOrientation(self._body_id)
        return Transform(orn, pos).toHomogeneousMatrix()

    @pose.setter
    def pose(self, pose_args):
        pose = parse_pose_args(pose_args)
        pos, orn = pose.translation, pose.quaternion.coeffs()
        self._client.resetBasePositionAndOrientation(self._body_id, pos, orn)

    def get_state(self):
        return dict(TWO=self.pose,
                    name=self.name,
                    scale=self._scale,
                    body_id=self._body_id)

    @property
    def visual_shape_data(self):
        return self._client.getVisualShapeData(self.body_id)

    @property
    def body_id(self):
        return self._body_id

    @property
    def client_id(self):
        return self.client_id

    @staticmethod
    def load(urdf_path, scale=1.0, client_id=0):
        urdf_path = Path(urdf_path)
        assert urdf_path.exists, 'URDF does not exist.'
        body_id = pb.loadURDF(urdf_path.as_posix(), physicsClientId=client_id, globalScaling=scale)
        return Body(body_id, scale=scale, client_id=client_id)
