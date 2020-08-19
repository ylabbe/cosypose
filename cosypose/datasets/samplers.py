import torch
import numpy as np
from torch.utils.data import Sampler
from cosypose.utils.random import temp_numpy_seed


class PartialSampler(Sampler):
    def __init__(self, ds, epoch_size):
        self.n_items = len(ds)
        self.epoch_size = min(epoch_size, len(ds))
        super().__init__(None)

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return (i.item() for i in torch.randperm(self.n_items)[:len(self)])


class DistributedSceneSampler(Sampler):
    def __init__(self, scene_ds, num_replicas, rank, shuffle=True):
        indices = np.arange(len(scene_ds))
        if shuffle:
            with temp_numpy_seed(0):
                indices = np.random.permutation(indices)
        all_indices = np.array_split(indices, num_replicas)
        local_indices = all_indices[rank].tolist()
        self.local_indices = local_indices

    def __len__(self):
        return len(self.local_indices)

    def __iter__(self):
        return iter(self.local_indices)


class ListSampler(Sampler):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)
