class SceneDatasetWrapper:
    def __init__(self, scene_ds):
        self.scene_ds = scene_ds

    @property
    def unwrapped(self):
        if isinstance(self, SceneDatasetWrapper):
            return self.scene_ds
        else:
            return self

    def __len__(self):
        return len(self.scene_ds)

    def __getitem__(self, idx):
        data = self.scene_ds[idx]
        return self.process_data(data)

    def process_data(self, data):
        return data
