import numpy as np
import trimesh
import torch
from copy import deepcopy

from .mesh_ops import get_meshes_bounding_boxes, sample_points
from .symmetries import make_bop_symmetries
from cosypose.utils.tensor_collection import TensorCollection


class MeshDataBase:
    def __init__(self, obj_list):
        self.infos = {obj['label']: obj for obj in obj_list}
        self.meshes = {l: trimesh.load(obj['mesh_path']) for l, obj in self.infos.items()}

    @staticmethod
    def from_object_ds(object_ds):
        obj_list = [object_ds[n] for n in range(len(object_ds))]
        return MeshDataBase(obj_list)

    def batched(self, aabb=False, resample_n_points=None, n_sym=64):
        if aabb:
            assert resample_n_points is None

        labels, points, symmetries = [], [], []
        new_infos = deepcopy(self.infos)
        for label, mesh in self.meshes.items():
            if aabb:
                points_n = get_meshes_bounding_boxes(torch.as_tensor(mesh.vertices).unsqueeze(0))[0]
            elif resample_n_points:
                points_n = torch.tensor(trimesh.sample.sample_surface(mesh, resample_n_points)[0])
            else:
                points_n = torch.tensor(mesh.vertices)
            points_n = points_n.clone()
            infos = self.infos[label]
            if infos['mesh_units'] == 'mm':
                scale = 0.001
            elif infos['mesh_units'] == 'm':
                scale = 1.0
            else:
                raise ValueError('Unit not supported', infos['mesh_units'])
            points_n *= scale

            dict_symmetries = {k: infos.get(k, []) for k in ('symmetries_discrete', 'symmetries_continuous')}
            symmetries_n = make_bop_symmetries(dict_symmetries, n_symmetries_continuous=n_sym, scale=scale)

            new_infos[label]['n_points'] = points_n.shape[0]
            new_infos[label]['n_sym'] = symmetries_n.shape[0]
            symmetries.append(torch.as_tensor(symmetries_n))
            points.append(torch.as_tensor(points_n))
            labels.append(label)

        labels = np.array(labels)
        points = pad_stack_tensors(points, fill='select_random', deterministic=True)
        symmetries = pad_stack_tensors(symmetries, fill=torch.eye(4), deterministic=True)
        return BatchedMeshes(new_infos, labels, points, symmetries).float()


class BatchedMeshes(TensorCollection):
    def __init__(self, infos, labels, points, symmetries):
        super().__init__()
        self.infos = infos
        self.label_to_id = {label: n for n, label in enumerate(labels)}
        self.labels = np.asarray(labels)
        self.register_tensor('points', points)
        self.register_tensor('symmetries', symmetries)

    @property
    def n_sym_mapping(self):
        return {label: obj['n_sym'] for label, obj in self.infos.items()}

    def select(self, labels):
        ids = [self.label_to_id[l] for l in labels]
        return Meshes(
            infos=[self.infos[l] for l in labels],
            labels=self.labels[ids],
            points=self.points[ids],
            symmetries=self.symmetries[ids],
        )


class Meshes(TensorCollection):
    def __init__(self, infos, labels, points, symmetries):
        super().__init__()
        self.infos = infos
        self.labels = np.asarray(labels)
        self.register_tensor('points', points)
        self.register_tensor('symmetries', symmetries)

    def select_labels(self, labels):
        raise NotImplementedError

    def sample_points(self, n_points, deterministic=False):
        return sample_points(self.points, n_points, deterministic=deterministic)


def pad_stack_tensors(tensor_list, fill='select_random', deterministic=True):
    n_max = max([t.shape[0] for t in tensor_list])
    if deterministic:
        np_random = np.random.RandomState(0)
    else:
        np_random = np.random
    tensor_list_padded = []
    for tensor_n in tensor_list:
        n_pad = n_max - len(tensor_n)

        if n_pad > 0:
            if isinstance(fill, torch.Tensor):
                assert isinstance(fill, torch.Tensor)
                assert fill.shape == tensor_n.shape[1:]
                pad = fill.unsqueeze(0).repeat(n_pad, *[1 for _ in fill.shape]).to(tensor_n.device).to(tensor_n.dtype)
            else:
                assert fill == 'select_random'
                ids_pad = np_random.choice(np.arange(len(tensor_n)), size=n_pad)
                pad = tensor_n[ids_pad]
            tensor_n_padded = torch.cat((tensor_n, pad), dim=0)
        else:
            tensor_n_padded = tensor_n
        tensor_list_padded.append(tensor_n_padded)
    return torch.stack(tensor_list_padded)
