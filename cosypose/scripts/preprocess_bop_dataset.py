from tqdm import tqdm
from PIL import Image
from cosypose.datasets.datasets_cfg import make_scene_dataset

if __name__ == '__main__':
    ds_name = 'itodd.pbr'

    scene_ds = make_scene_dataset(ds_name)
    for n in tqdm(range(len(scene_ds))):
        rgb, mask, state = scene_ds[n]
        row = state['frame_info']
        scene_id, view_id = row['scene_id'], row['view_id']
        view_id = int(view_id)
        view_id_str = f'{view_id:06d}'
        scene_id_str = f'{int(scene_id):06d}'

        scene_dir = scene_ds.base_dir / scene_id_str
        p = scene_dir / 'mask_visib' / f'{view_id_str}_all.png'
        Image.fromarray(mask.numpy()).save(p)
