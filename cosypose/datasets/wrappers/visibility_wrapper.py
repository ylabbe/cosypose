import numpy as np
from .base import SceneDatasetWrapper


class VisibilityWrapper(SceneDatasetWrapper):
    def process_data(self, data):
        rgb, mask, state = data
        ids_visible = np.unique(mask)
        ids_visible = set(ids_visible[ids_visible > 0])
        visib_objects = []
        for obj in state['objects']:
            if obj['id_in_segm'] in ids_visible:
                visib_objects.append(obj)
        state['objects'] = visib_objects
        return rgb, mask, state
