import numpy as np
import pandas as pd

from cosypose.config import LOCAL_DATA_DIR, ASSET_DIR, BOP_DS_DIR
from cosypose.utils.logging import get_logger

from .bop_object_datasets import BOPObjectDataset
from .bop import BOPDataset, remap_bop_targets
from .urdf_dataset import BOPUrdfDataset, OneUrdfDataset
from .texture_dataset import TextureDataset


logger = get_logger(__name__)


def _make_tless_dataset(split):
    ds_dir = BOP_DS_DIR / 'tless'
    ds = BOPDataset(ds_dir, split=split)
    return ds


def keep_bop19(ds):
    targets = pd.read_json(ds.ds_dir / 'test_targets_bop19.json')
    targets = remap_bop_targets(targets)
    targets = targets.loc[:, ['scene_id', 'view_id']].drop_duplicates()
    index = ds.frame_index.merge(targets, on=['scene_id', 'view_id']).reset_index(drop=True)
    assert len(index) == len(targets)
    ds.frame_index = index
    return ds


def make_scene_dataset(ds_name, n_frames=None):
    # TLESS
    if ds_name == 'tless.primesense.train':
        ds = _make_tless_dataset('train_primesense')

    elif ds_name == 'tless.primesense.test':
        ds = _make_tless_dataset('test_primesense')

    elif ds_name == 'tless.primesense.test.bop19':
        ds = _make_tless_dataset('test_primesense')
        ds = keep_bop19(ds)

    # YCBV
    elif ds_name == 'ycbv.train.real':
        ds_dir = BOP_DS_DIR / 'ycbv'
        ds = BOPDataset(ds_dir, split='train_real')

    elif ds_name == 'ycbv.train.synt':
        ds_dir = BOP_DS_DIR / 'ycbv'
        ds = BOPDataset(ds_dir, split='train_synt')

    elif ds_name == 'ycbv.test':
        ds_dir = BOP_DS_DIR / 'ycbv'
        ds = BOPDataset(ds_dir, split='test')

    elif ds_name == 'ycbv.test.keyframes':
        ds_dir = BOP_DS_DIR / 'ycbv'
        ds = BOPDataset(ds_dir, split='test')
        keyframes_path = ds_dir / 'keyframe.txt'
        ls = keyframes_path.read_text().split('\n')[:-1]
        frame_index = ds.frame_index
        ids = []
        for l_n in ls:
            scene_id, view_id = l_n.split('/')
            scene_id, view_id = int(scene_id), int(view_id)
            mask = (frame_index['scene_id'] == scene_id) & (frame_index['view_id'] == view_id)
            ids.append(np.where(mask)[0].item())
        ds.frame_index = frame_index.iloc[ids].reset_index(drop=True)

    # BOP challenge
    elif ds_name == 'hb.bop19':
        ds_dir = BOP_DS_DIR / 'hb'
        ds = BOPDataset(ds_dir, split='test_primesense')
        ds = keep_bop19(ds)
    elif ds_name == 'icbin.bop19':
        ds_dir = BOP_DS_DIR / 'icbin'
        ds = BOPDataset(ds_dir, split='test')
        ds = keep_bop19(ds)
    elif ds_name == 'itodd.bop19':
        ds_dir = BOP_DS_DIR / 'itodd'
        ds = BOPDataset(ds_dir, split='test')
        ds = keep_bop19(ds)
    elif ds_name == 'lmo.bop19':
        ds_dir = BOP_DS_DIR / 'lmo'
        ds = BOPDataset(ds_dir, split='test')
        ds = keep_bop19(ds)
    elif ds_name == 'tless.bop19':
        ds_dir = BOP_DS_DIR / 'tless'
        ds = BOPDataset(ds_dir, split='test_primesense')
        ds = keep_bop19(ds)
    elif ds_name == 'tudl.bop19':
        ds_dir = BOP_DS_DIR / 'tudl'
        ds = BOPDataset(ds_dir, split='test')
        ds = keep_bop19(ds)
    elif ds_name == 'ycbv.bop19':
        ds_dir = BOP_DS_DIR / 'ycbv'
        ds = BOPDataset(ds_dir, split='test')
        ds = keep_bop19(ds)

    elif ds_name == 'hb.pbr':
        ds_dir = BOP_DS_DIR / 'hb'
        ds = BOPDataset(ds_dir, split='train_pbr')
    elif ds_name == 'icbin.pbr':
        ds_dir = BOP_DS_DIR / 'icbin'
        ds = BOPDataset(ds_dir, split='train_pbr')
    elif ds_name == 'itodd.pbr':
        ds_dir = BOP_DS_DIR / 'itodd'
        ds = BOPDataset(ds_dir, split='train_pbr')
    elif ds_name == 'lm.pbr':
        ds_dir = BOP_DS_DIR / 'lm'
        ds = BOPDataset(ds_dir, split='train_pbr')
    elif ds_name == 'tless.pbr':
        ds_dir = BOP_DS_DIR / 'tless'
        ds = BOPDataset(ds_dir, split='train_pbr')
    elif ds_name == 'tudl.pbr':
        ds_dir = BOP_DS_DIR / 'tudl'
        ds = BOPDataset(ds_dir, split='train_pbr')
    elif ds_name == 'ycbv.pbr':
        ds_dir = BOP_DS_DIR / 'ycbv'
        ds = BOPDataset(ds_dir, split='train_pbr')

    elif ds_name == 'hb.val':
        ds_dir = BOP_DS_DIR / 'hb'
        ds = BOPDataset(ds_dir, split='val_primesense')
    elif ds_name == 'itodd.val':
        ds_dir = BOP_DS_DIR / 'itodd'
        ds = BOPDataset(ds_dir, split='val')
    elif ds_name == 'tudl.train.real':
        ds_dir = BOP_DS_DIR / 'tudl'
        ds = BOPDataset(ds_dir, split='train_real')

    # Synthetic datasets
    elif 'synthetic.' in ds_name:
        from .synthetic_dataset import SyntheticSceneDataset
        assert '.train' in ds_name or '.val' in ds_name
        is_train = 'train' in ds_name.split('.')[-1]
        ds_name = ds_name.split('.')[1]
        ds = SyntheticSceneDataset(ds_dir=LOCAL_DATA_DIR / 'synt_datasets' / ds_name, train=is_train)

    else:
        raise ValueError(ds_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_object_dataset(ds_name):
    ds = None
    if ds_name == 'tless.cad':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    elif ds_name == 'tless.eval' or ds_name == 'tless.bop':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_eval')

    # YCBV
    elif ds_name == 'ycbv.bop':
        ds = BOPObjectDataset(BOP_DS_DIR / 'ycbv/models')
    elif ds_name == 'ycbv.bop-compat':
        # BOP meshes (with their offsets) and symmetries
        # Replace symmetries of objects not considered symmetric in PoseCNN
        ds = BOPObjectDataset(BOP_DS_DIR / 'ycbv/models_bop-compat')
    elif ds_name == 'ycbv.bop-compat.eval':
        # PoseCNN eval meshes and symmetries, WITH bop offsets
        ds = BOPObjectDataset(BOP_DS_DIR / 'ycbv/models_bop-compat_eval')

    # Other BOP
    elif ds_name == 'hb':
        ds = BOPObjectDataset(BOP_DS_DIR / 'hb/models')
    elif ds_name == 'icbin':
        ds = BOPObjectDataset(BOP_DS_DIR / 'icbin/models')
    elif ds_name == 'itodd':
        ds = BOPObjectDataset(BOP_DS_DIR / 'itodd/models')
    elif ds_name == 'lm':
        ds = BOPObjectDataset(BOP_DS_DIR / 'lm/models')
    elif ds_name == 'tudl':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tudl/models')

    else:
        raise ValueError(ds_name)
    return ds


def make_urdf_dataset(ds_name):
    if isinstance(ds_name, list):
        ds_index = []
        for ds_name_n in ds_name:
            dataset = make_urdf_dataset(ds_name_n)
            ds_index.append(dataset.index)
        dataset.index = pd.concat(ds_index, axis=0)
        return dataset

    # BOP
    if ds_name == 'tless.cad':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tless.cad')
    elif ds_name == 'tless.reconst':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tless.reconst')
    elif ds_name == 'ycbv':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'ycbv')
    elif ds_name == 'hb':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'hb')
    elif ds_name == 'icbin':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'icbin')
    elif ds_name == 'itodd':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'itodd')
    elif ds_name == 'lm':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'lm')
    elif ds_name == 'tudl':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tudl')

    # Custom scenario
    elif 'custom' in ds_name:
        scenario = ds_name.split('.')[1]
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'scenarios' / scenario / 'urdfs')

    elif ds_name == 'camera':
        ds = OneUrdfDataset(ASSET_DIR / 'camera/model.urdf', 'camera')
    else:
        raise ValueError(ds_name)
    return ds


def make_texture_dataset(ds_name):
    if ds_name == 'shapenet':
        ds = TextureDataset(LOCAL_DATA_DIR / 'texture_datasets' / 'shapenet')
    else:
        raise ValueError(ds_name)
    return ds
