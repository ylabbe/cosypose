from .base import SceneDatasetWrapper


class AugmentationWrapper(SceneDatasetWrapper):
    def __init__(self, scene_ds, augmentation):
        self.augmentation = augmentation
        super().__init__(scene_ds)
        self.frame_index = self.scene_ds.frame_index

    def process_data(self, data):
        rgb, mask, obs = data
        return self.augmentation(rgb, mask, obs)
