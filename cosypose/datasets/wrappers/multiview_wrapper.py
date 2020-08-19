import pandas as pd
import torch
import numpy as np

from .base import SceneDatasetWrapper


class MultiViewWrapper(SceneDatasetWrapper):
    def __init__(self, scene_ds, n_views=4):

        n_max_views = n_views
        frame_index = scene_ds.frame_index.copy().reset_index(drop=True)
        groups = frame_index.groupby(['scene_id']).groups

        random_state = np.random.RandomState(0)
        self.frame_index = []
        for scene_id, group_ids in groups.items():
            n_max_views = n_views
            group_ids = random_state.permutation(group_ids)
            len_group = len(group_ids)
            for k, m in enumerate(np.arange(len_group)[::n_max_views]):
                ids_k = np.arange(len(group_ids))[m:m+n_max_views].tolist()
                ds_ids = group_ids[ids_k]
                df_group = frame_index.loc[ds_ids]
                self.frame_index.append(dict(
                    scene_id=scene_id,
                    view_ids=df_group['view_id'].values.tolist(),
                    n_views=len(df_group),
                    scene_ds_ids=ds_ids,
                ))

        self.frame_index = pd.DataFrame(self.frame_index)
        self.frame_index['group_id'] = np.arange(len(self.frame_index))
        self.scene_ds = scene_ds

    def __getitem__(self, idx):
        row = self.frame_index.iloc[idx]
        ds_ids = row['scene_ds_ids']
        rgbs, masks, obss = [], [], []
        for ds_id in ds_ids:
            rgb, mask, obs = self.scene_ds[ds_id]
            rgbs.append(rgb)
            masks.append(mask)
            obs['frame_info']['group_id'] = row['group_id']
            obss.append(obs)

        rgbs = torch.stack(rgbs)
        masks = torch.stack(masks)
        return rgbs, masks, obss

    def __len__(self):
        return len(self.frame_index)
