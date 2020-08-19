import numpy as np
from .body import Body
from copy import deepcopy
from collections import defaultdict
from .client import BulletClient


class BodyCache:
    def __init__(self, urdf_ds, client_id):
        self.urdf_ds = urdf_ds
        self.client = BulletClient(client_id)
        self.cache = defaultdict(list)
        self.away_transform = (0, 0, 1000), (0, 0, 0, 1)

    def _load_body(self, label):
        ds_idx = np.where(self.urdf_ds.index['label'] == label)[0].item()
        object_infos = self.urdf_ds[ds_idx].to_dict()
        body = Body.load(object_infos['urdf_path'],
                         scale=object_infos['scale'],
                         client_id=self.client.client_id)
        body.pose = self.away_transform
        self.cache[object_infos['label']].append(body)
        return body

    def hide_bodies(self):
        n = 0
        for body_list in self.cache.values():
            for body in body_list:
                pos = (1000, 1000, 1000 + n * 10)
                orn = (0, 0, 0, 1)
                body.pose = pos, orn
                n += 1

    def get_bodies_by_labels(self, labels):
        self.hide_bodies()
        gb_label = defaultdict(lambda: 0)
        for label in labels:
            gb_label[label] += 1

        for label, n_instances in gb_label.items():
            n_missing = gb_label[label] - len(self.cache[label])
            for n in range(n_missing):
                self._load_body(label)

        remaining = deepcopy(dict(self.cache))
        bodies = [remaining[label].pop(0) for label in labels]
        return bodies

    def get_bodies_by_ids(self, ids):
        labels = [self.urdf_ds[idx]['label'] for idx in ids]
        return self.get_bodies_by_labels(labels)

    def __len__(self):
        return sum([len(bodies) for bodies in self.cache.values()])


class TextureCache:
    def __init__(self, texture_ds, client_id):
        self.texture_ds = texture_ds
        self.client = BulletClient(client_id)
        self.cache = dict()

    def _load_texture(self, idx):
        self.cache[idx] = self.client.loadTexture(str(self.texture_ds[idx]['texture_path']))

    def get_texture(self, idx):
        if idx not in self.cache:
            self._load_texture(idx)
        return self.cache[idx]

    @property
    def cached_textures(self):
        return list(self.cache.values())
