import argparse
import subprocess
from tqdm import tqdm
import os
import numpy as np
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper
from cosypose.config import LOCAL_DATA_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running COLMAP')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nviews', type=int, default=4)
    args = parser.parse_args()

    assert args.nviews > 1
    if args.dataset == 'tless':
        scene_ds = make_scene_dataset('tless.primesense.test.bop19')
    elif args.dataset == 'ycbv':
        scene_ds = make_scene_dataset('ycbv.test.keyframes')
    else:
        raise ValueError
    scene_ds = MultiViewWrapper(scene_ds, n_views=args.nviews)

    colmap_dir = LOCAL_DATA_DIR / 'colmap' / f'{args.dataset}_nviews={args.nviews}'
    colmap_dir.mkdir(exist_ok=True, parents=True)

    def path_to_im(scene_id, view_id):
        scene = f'{int(scene_id):06d}'
        view = f'{int(view_id):06d}.png'
        path = scene_ds.unwrapped.base_dir / scene / 'rgb' / view
        return path

    for group_id, group in tqdm(scene_ds.frame_index.groupby('group_id')):
        view_ids = group['view_ids'].values[0]
        scene_id = np.unique(group['scene_id']).item()
        views_str = '-'.join(map(str, view_ids))
        group_dir = colmap_dir / f'{args.dataset}_groupid={group_id}_scene={scene_id}-views={views_str}'
        group_images_dir = group_dir / 'images'
        group_dir.mkdir(exist_ok=True)
        group_images_dir.mkdir(exist_ok=True)
        for view_id in view_ids:
            ds_im_path = path_to_im(scene_id, view_id)
            try:
                os.symlink(ds_im_path, group_images_dir / ds_im_path.name)
            except FileExistsError:
                pass

        colmap_ds_path = group_dir
        cmd = ['colmap', 'automatic_reconstructor',
               '--workspace_path', colmap_ds_path.as_posix(),
               '--image_path', (colmap_ds_path / 'images').as_posix()]
        print(group_dir)
        subprocess.run(cmd)
