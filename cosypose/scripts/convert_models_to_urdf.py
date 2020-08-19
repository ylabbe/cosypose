from pathlib import Path
import argparse
import shutil
from cosypose.config import LOCAL_DATA_DIR
from tqdm import tqdm

from cosypose.datasets.datasets_cfg import make_object_dataset
from cosypose.libmesh import ply_to_obj, obj_to_urdf
from cosypose.libmesh import downsample_obj


def convert_obj_dataset_to_urdfs(obj_ds_name, texture_size=(1024, 1024), n_faces=None):
    obj_dataset = make_object_dataset(obj_ds_name)
    urdf_dir = LOCAL_DATA_DIR / 'urdfs' / obj_ds_name
    urdf_dir.mkdir(exist_ok=True, parents=True)
    for n in tqdm(range(len(obj_dataset))):
        obj = obj_dataset[n]
        ply_path = Path(obj['mesh_path'])
        out_dir = urdf_dir / obj['label']
        out_dir.mkdir(exist_ok=True)
        obj_path = out_dir / ply_path.with_suffix('.obj').name
        ply_to_obj(ply_path, obj_path, texture_size=texture_size)

        if n_faces is not None:
            downsample_path = obj_path.parent / 'downsample.obj'
            downsample_obj(obj_path, downsample_path, n_faces=n_faces)
            shutil.copy(downsample_path, obj_path)

        obj_to_urdf(obj_path, obj_path.with_suffix('.urdf'))


def main():
    parser = argparse.ArgumentParser('3D ply object models -> pybullet URDF converter')
    parser.add_argument('--models', default='', type=str)
    args = parser.parse_args()
    convert_obj_dataset_to_urdfs(args.models)


if __name__ == '__main__':
    main()
