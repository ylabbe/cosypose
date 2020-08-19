from sklearn.metrics import average_precision_score
import numpy as np
import xarray as xr
from collections import OrderedDict
import torch
from torch.utils.data import TensorDataset, DataLoader

from cosypose.lib3d.distances import dists_add, dists_add_symmetric
from cosypose.utils.xarray import xr_merge

from .utils import (match_poses, get_top_n_ids,
                    add_valid_gt, get_candidate_matches, add_inst_num,
                    compute_auc_posecnn)
from .base import Meter


class PoseErrorMeter(Meter):
    def __init__(self,
                 mesh_db,
                 error_type='ADD',
                 report_AP=False,
                 report_error_AUC=False,
                 report_error_stats=False,
                 sample_n_points=None,
                 errors_bsz=1,
                 match_threshold=0.1,
                 exact_meshes=True,
                 spheres_overlap_check=True,
                 consider_all_predictions=False,
                 targets=None,
                 visib_gt_min=-1,
                 n_top=-1):

        self.sample_n_points = sample_n_points
        self.mesh_db = mesh_db.batched().cuda().float()
        self.error_type = error_type.upper()
        self.errors_bsz = errors_bsz
        self.n_top = n_top
        self.exact_meshes = exact_meshes
        self.visib_gt_min = visib_gt_min
        self.targets = targets
        self.match_threshold = match_threshold
        self.spheres_overlap_check = spheres_overlap_check
        self.consider_all_predictions = consider_all_predictions
        self.report_AP = report_AP
        self.report_error_stats = report_error_stats
        self.report_error_AUC = report_error_AUC
        self.reset()

        if self.exact_meshes:
            assert self.errors_bsz == 1 and sample_n_points is None

    def compute_errors(self, TXO_pred, TXO_gt, labels):
        meshes = self.mesh_db.select(labels)

        if self.exact_meshes:
            assert len(labels) == 1
            n_points = self.mesh_db.infos[labels[0]]['n_points']
            points = meshes.points[:, :n_points]
        else:
            if self.sample_n_points is not None:
                points = meshes.sample_points(self.sample_n_points, deterministic=True)
            else:
                points = meshes.points

        if self.error_type.upper() == 'ADD':
            dists = dists_add(TXO_pred, TXO_gt, points)

        elif self.error_type.upper() == 'ADD-S':
            dists = dists_add_symmetric(TXO_pred, TXO_gt, points)

        elif self.error_type.upper() == 'ADD(-S)':
            ids_nosym, ids_sym = [], []
            for n, label in enumerate(labels):
                if self.mesh_db.infos[label]['is_symmetric']:
                    ids_sym.append(n)
                else:
                    ids_nosym.append(n)
            dists = torch.empty((len(TXO_pred), points.shape[1], 3), dtype=TXO_pred.dtype, device=TXO_pred.device)
            if len(ids_nosym) > 0:
                dists[ids_nosym] = dists_add(TXO_pred[ids_nosym], TXO_gt[ids_nosym], points[ids_nosym])
            if len(ids_sym) > 0:
                dists[ids_sym] = dists_add_symmetric(TXO_pred[ids_sym], TXO_gt[ids_sym], points[ids_sym])
        else:
            raise ValueError("Error not supported", self.error_type)

        errors = dict()
        errors['norm_avg'] = torch.norm(dists, dim=-1, p=2).mean(-1)
        errors['xyz_avg'] = dists.abs().mean(dim=-2)
        errors['TCO_xyz'] = (TXO_pred[:, :3, -1] - TXO_gt[:, :3, -1]).abs()
        errors['TCO_norm'] = torch.norm(TXO_pred[:, :3, -1] - TXO_gt[:, :3, -1], dim=-1, p=2)
        return errors

    def compute_errors_batch(self, TXO_pred, TXO_gt, labels):
        errors = []
        ids = torch.arange(len(labels))
        ds = TensorDataset(TXO_pred, TXO_gt, ids)
        dl = DataLoader(ds, batch_size=self.errors_bsz)
        for (TXO_pred_, TXO_gt_, ids_) in dl:
            labels_ = labels[ids_.numpy()]
            errors.append(self.compute_errors(TXO_pred_, TXO_gt_, labels_))

        if len(errors) == 0:
            errors.append(dict(
                norm_avg=torch.empty(0, dtype=torch.float),
                xyz_avg=torch.empty(0, 3, dtype=torch.float),
                TCO_xyz=torch.empty((0, 3), dtype=torch.float),
                TCO_norm=torch.empty(0, dtype=torch.float),
            ))

        errorsd = dict()
        for k in errors[0].keys():
            errorsd[k] = torch.cat([errors_n[k] for errors_n in errors], dim=0)
        return errorsd

    def add(self, pred_data, gt_data):
        group_keys = ['scene_id', 'view_id', 'label']

        pred_data = pred_data.float()
        gt_data = gt_data.float()

        # Only keep predictions relevant to gt scene and images.
        gt_infos = gt_data.infos.loc[:, ['scene_id', 'view_id']].drop_duplicates().reset_index(drop=True)
        targets = self.targets
        if targets is not None:
            targets = gt_infos.merge(targets)
        pred_data.infos['batch_pred_id'] = np.arange(len(pred_data))
        keep_ids = gt_infos.merge(pred_data.infos)['batch_pred_id']
        pred_data = pred_data[keep_ids]

        # Add inst id to the dataframes
        pred_data.infos = add_inst_num(pred_data.infos, key='pred_inst_id', group_keys=group_keys)
        gt_data.infos = add_inst_num(gt_data.infos, key='gt_inst_id', group_keys=group_keys)

        # Filter predictions according to BOP evaluation.
        if not self.consider_all_predictions:
            ids_top_n_pred = get_top_n_ids(pred_data.infos,
                                           group_keys=group_keys, top_key='score',
                                           targets=targets, n_top=self.n_top)
            pred_data_filtered = pred_data.clone()[ids_top_n_pred]
        else:
            pred_data_filtered = pred_data.clone()

        # Compute valid targets according to BOP evaluation.
        gt_data.infos = add_valid_gt(gt_data.infos,
                                     group_keys=group_keys,
                                     targets=targets,
                                     visib_gt_min=self.visib_gt_min)

        # Compute tentative candidates
        cand_infos = get_candidate_matches(pred_data_filtered.infos, gt_data.infos,
                                           group_keys=group_keys,
                                           only_valids=True)

        # Filter out tentative matches that are too far.
        if self.spheres_overlap_check:
            diameters = [self.mesh_db.infos[k]['diameter_m'] for k in cand_infos['label']]
            dists = pred_data_filtered[cand_infos['pred_id'].values.tolist()].poses[:, :3, -1] - \
                gt_data[cand_infos['gt_id'].values.tolist()].poses[:, :3, -1]
            spheres_overlap = torch.norm(dists, dim=-1) < torch.as_tensor(diameters).to(dists.dtype).to(dists.device)
            keep_ids = np.where(spheres_overlap.cpu().numpy())[0]
            cand_infos = cand_infos.iloc[keep_ids].reset_index(drop=True)
            cand_infos['cand_id'] = np.arange(len(cand_infos))

        pred_ids = cand_infos['pred_id'].values.tolist()
        gt_ids = cand_infos['gt_id'].values.tolist()
        cand_TXO_gt = gt_data.poses[gt_ids]
        cand_TXO_pred = pred_data_filtered.poses[pred_ids]

        # Compute errors for tentative matches
        errors = self.compute_errors_batch(cand_TXO_pred, cand_TXO_gt,
                                           cand_infos['label'].values)

        # Matches can only be objects within thresholds (following BOP).
        cand_infos['error'] = errors['norm_avg'].cpu().numpy()
        cand_infos['obj_diameter'] = [self.mesh_db.infos[k]['diameter_m'] for k in cand_infos['label']]
        keep = cand_infos['error'] <= self.match_threshold * cand_infos['obj_diameter']
        cand_infos = cand_infos[keep].reset_index(drop=True)

        # Match predictions to ground truth poses
        matches = match_poses(cand_infos, group_keys=group_keys)

        # Save all informations in xarray datasets
        gt_keys = group_keys + ['gt_inst_id', 'valid'] + (['visib_fract'] if 'visib_fract' in gt_infos else [])
        gt = gt_data.infos.loc[:, gt_keys]
        preds = pred_data.infos.loc[:, group_keys + ['pred_inst_id', 'score']]
        matches = matches.loc[:, group_keys + ['pred_inst_id', 'gt_inst_id', 'cand_id']]

        gt = xr.Dataset(gt).rename({'dim_0': 'gt_id'})
        matches = xr.Dataset(matches).rename({'dim_0': 'match_id'})
        preds = xr.Dataset(preds).rename({'dim_0': 'pred_id'})

        errors_norm = errors['norm_avg'].cpu().numpy()[matches['cand_id'].values]
        errors_xyz = errors['xyz_avg'].cpu().numpy()[matches['cand_id'].values]
        errors_TCO_xyz = errors['TCO_xyz'].cpu().numpy()[matches['cand_id'].values]
        errors_TCO_norm = errors['TCO_norm'].cpu().numpy()[matches['cand_id'].values]

        matches['obj_diameter'] = 'match_id', [self.mesh_db.infos[k.item()]['diameter_m'] for k in matches['label']]
        matches['norm'] = 'match_id', errors_norm
        matches['0.1d'] = 'match_id', errors_norm < 0.1 * matches['obj_diameter']
        matches['xyz'] = ('match_id', 'dim3'), errors_xyz
        matches['TCO_xyz'] = ('match_id', 'dim3'), errors_TCO_xyz
        matches['TCO_norm'] = 'match_id', errors_TCO_norm

        preds['TXO_pred'] = ('pred_id', 'Trow', 'Tcol'), pred_data.poses.cpu().numpy()

        fill_values = {
            'norm': np.inf,
            '0.1d': False,
            'xyz': np.inf,
            'TCO_xyz': np.inf,
            'TCO_norm': np.inf,
            'obj_diameter': np.nan,
            'TXO_pred': np.nan,
            'score': np.nan,
        }
        matches = xr_merge(matches, preds, on=group_keys + ['pred_inst_id'],
                           dim1='match_id', dim2='pred_id', fill_value=fill_values)
        gt = xr_merge(gt, matches, on=group_keys + ['gt_inst_id'],
                      dim1='gt_id', dim2='match_id', fill_value=fill_values)

        preds_match_merge = xr_merge(preds, matches, on=group_keys+['pred_inst_id'],
                                     dim1='pred_id', dim2='match_id', fill_value=fill_values)
        preds['0.1d'] = 'pred_id', preds_match_merge['0.1d']

        self.datas['gt_df'].append(gt)
        self.datas['pred_df'].append(preds)
        self.datas['matches_df'].append(matches)

    def summary(self):
        gt_df = xr.concat(self.datas['gt_df'], dim='gt_id')
        matches_df = xr.concat(self.datas['matches_df'], dim='match_id')
        pred_df = xr.concat(self.datas['pred_df'], dim='pred_id')

        # ADD-S AUC
        valid_df = gt_df.sel(gt_id=gt_df['valid'])
        AUC = OrderedDict()
        for (label, ids) in valid_df.groupby('label').groups.items():
            errors = valid_df['norm'].values[ids]
            assert np.all(~np.isnan(errors))
            AUC[label] = compute_auc_posecnn(errors)
        gt_df['AUC/objects'] = xr.DataArray(
            list(AUC.values()), [('objects', list(AUC.keys()))], dims=['objects'])
        gt_df['AUC/objects/mean'] = gt_df['AUC/objects'].mean('objects')
        gt_df['AUC'] = compute_auc_posecnn(valid_df['norm'])

        # AP/mAP@0.1d
        valid_k = '0.1d'
        n_gts = dict()

        if self.n_top > 0:
            group_keys = ['scene_id', 'view_id', 'label']
            subdf = gt_df[[*group_keys, 'valid']].to_dataframe().groupby(group_keys).sum().reset_index()
            subdf['gt_count'] = np.minimum(self.n_top, subdf['valid'])
            for label, group in subdf.groupby('label'):
                n_gts[label] = group['gt_count'].sum()
        else:
            subdf = gt_df[['label', 'valid']].groupby('label').sum()
            for label in subdf['label'].values:
                n_gts[label] = subdf.sel(label=label)['valid'].item()

        ap_dfs = dict()

        def compute_ap(label_df, label_n_gt):
            label_df = label_df.sort_values('score', ascending=False).reset_index(drop=True)
            label_df['n_tp'] = np.cumsum(label_df[valid_k].values.astype(np.float))
            label_df['prec'] = label_df['n_tp'] / (np.arange(len(label_df)) + 1)
            label_df['recall'] = label_df['n_tp'] / label_n_gt
            y_true = label_df[valid_k]
            y_score = label_df['score']
            ap = average_precision_score(y_true, y_score) * y_true.sum() / label_n_gt
            label_df['AP'] = ap
            label_df['n_gt'] = label_n_gt
            return ap, label_df

        df = pred_df[['label', valid_k, 'score']].to_dataframe().set_index(['label'])
        for label, label_n_gt in n_gts.items():
            if df.index.contains(label):
                label_df = df.loc[[label]]
                if label_df[valid_k].sum() > 0:
                    ap, label_df = compute_ap(label_df, label_n_gt)
                    ap_dfs[label] = label_df

        if len(ap_dfs) > 0:
            mAP = np.mean([np.unique(ap_df['AP']).item() for ap_df in ap_dfs.values()])
            AP, ap_dfs['all'] = compute_ap(df.reset_index(), sum(list(n_gts.values())))
        else:
            AP, mAP = 0., 0.
        n_gt_valid = int(sum(list(n_gts.values())))

        summary = {
            'n_gt': len(gt_df['gt_id']),
            'n_gt_valid': n_gt_valid,
            'n_pred': len(pred_df['pred_id']),
            'n_matched': len(matches_df['match_id']),
            'matched_gt_ratio': len(matches_df['match_id']) / n_gt_valid,
            'pred_matched_ratio': len(pred_df['pred_id']) / max(len(matches_df['match_id']), 1),
            '0.1d': valid_df['0.1d'].sum('gt_id').item() / n_gt_valid,
        }

        if self.report_error_stats:
            summary.update({
                'norm': matches_df['norm'].mean('match_id').item(),
                'xyz': matches_df['xyz'].mean('match_id').values.tolist(),
                'TCO_xyz': matches_df['TCO_xyz'].mean('match_id').values.tolist(),
                'TCO_norm': matches_df['TCO_norm'].mean('match_id').values.tolist(),
            })

        if self.report_AP:
            summary.update({
                'AP': AP,
                'mAP': mAP,
            })

        if self.report_error_AUC:
            summary.update({
                'AUC/objects/mean': gt_df['AUC/objects/mean'].item(),
                'AUC': gt_df['AUC'].item(),
            })

        dfs = dict(gt=gt_df, matches=matches_df, preds=pred_df, ap=ap_dfs)
        return summary, dfs
