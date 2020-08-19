from torch.utils.data import DataLoader
from tqdm import tqdm
from cosypose.datasets.pose_dataset import PoseDataset
from cosypose.datasets.datasets_cfg import make_scene_dataset

if __name__ == '__main__':
    from cosypose.bop_config import BOP_CONFIG
    for ds_name, bop_config in BOP_CONFIG.items():
        train_synt_real_ds_names = bop_config.get('train_synt_real_ds_names', [])
        for (ds_name_, _) in train_synt_real_ds_names:
            scene_ds = make_scene_dataset(ds_name_)
            print(scene_ds.name, len(scene_ds))

    # ds_name = 'dream.baxter.synt.dr.train'
    # ds_name = 'tudl.pbr'
    ds_name = 'tudl.pbr'

    scene_ds_train = make_scene_dataset(ds_name)
    ds_kwargs = dict(
        # resize=(480, 640),
        resize=(1280, 960),
        rgb_augmentation=False,
        background_augmentation=False,
    )
    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_iter_train = DataLoader(ds_train, shuffle=True, batch_size=32,
                               num_workers=8, collate_fn=ds_train.collate_fn,
                               drop_last=False, pin_memory=True)
    # ds_train[8129]

    for _ in range(1):
        for data in tqdm(ds_iter_train):
            pass
