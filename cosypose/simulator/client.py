import functools
import pybullet as pb


class BulletClient:
    def __init__(self, client_id):
        self.client_id = client_id

    def __getattr__(self, name):
        attribute = getattr(pb, name)
        attribute = functools.partial(attribute, physicsClientId=self.client_id)
        return attribute
