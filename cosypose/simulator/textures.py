import numpy as np
from collections import defaultdict
import pybullet as pb


def apply_random_textures(body, texture_ids, per_link=False, np_random=np.random):
    data = body.visual_shape_data
    visual_shapes_ids = [t[1] for t in data]
    n_shapes = defaultdict(lambda: 0)
    for i in visual_shapes_ids:
        n_shapes[i] += 1

    for link_id, n_shapes in n_shapes.items():
        texture_id = np_random.choice(texture_ids)
        for link_shape_id in range(n_shapes):
            if per_link:
                texture_id = np_random.choice(texture_ids)
            specular = np_random.randint(0, 1000)
            pb.changeVisualShape(body._body_id, link_id, link_shape_id,
                                 textureUniqueId=texture_id, rgbaColor=[1, 1, 1, 1],
                                 physicsClientId=body._client.client_id,
                                 specularColor=specular * np.ones(3))
    return
