import trimesh
import shutil
from copy import deepcopy
import numpy as np
import json
from cosypose.config import LOCAL_DATA_DIR


if __name__ == '__main__':
    ds_dir = LOCAL_DATA_DIR / 'bop_datasets/ycbv'
    models_dir = ds_dir / 'models'

    orig_names = (ds_dir / 'ycbv_friendly_names.txt').read_text()
    orig_names = {str(int(l.split(' ')[0])): l.split(' ')[1] for l in orig_names.split('\n')[:-1]}

    infos = json.loads((models_dir / 'models_info.json').read_text())
    compat_infos = deepcopy(infos)

    # Consider these 2 objects asymmetric
    for str_obj_id, orig_name in orig_names.items():
        if orig_name == '002_master_chef_can' or orig_name == '040_large_marker':
            compat_infos[str_obj_id]['symmetries_discrete'] = []
            compat_infos[str_obj_id]['symmetries_continuous'] = []

    bop_compat_dir = ds_dir / 'models_bop-compat'
    bop_compat_dir.mkdir(exist_ok=True)
    for file_path in models_dir.iterdir():
        shutil.copy(file_path, bop_compat_dir / file_path.name)
    (bop_compat_dir / 'models_info.json').write_text(json.dumps(compat_infos))

    l_offsets = (ds_dir / 'offsets.txt').read_text().split('\n')[:-1]
    offsets = dict()
    for l_n in l_offsets:
        obj_id, offset = l_n[:2], l_n[3:]
        obj_id = int(obj_id)
        offset = np.array(json.loads(offset))
        offsets[str(obj_id)] = offset

    # Models used in the original evaluation
    bop_compat_eval_dir = ds_dir / 'models_bop-compat_eval'
    bop_compat_eval_dir.mkdir(exist_ok=True)
    (bop_compat_eval_dir / 'models_info.json').write_text(json.dumps(compat_infos))
    for obj_id, orig_name in orig_names.items():
        xyz = (ds_dir / 'models_original' / orig_name / 'points.xyz').read_text()
        xyz = xyz.split('\n')[:-1]
        xyz = [list(map(float, xyz_n.split(' '))) for xyz_n in xyz]
        vertices = np.array(xyz) * 1000 + offsets[obj_id]
        mesh = trimesh.Trimesh(vertices=vertices)
        mesh.export(bop_compat_eval_dir / f'obj_{int(obj_id):06d}.ply')
