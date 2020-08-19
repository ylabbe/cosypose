import numpy as np
from sklearn.metrics import average_precision_score
import xarray as xr
import torchvision
import torch
from torch.utils.data import TensorDataset, DataLoader
from .base import Meter

from .utils import (match_poses, get_top_n_ids,
                    add_valid_gt, get_candidate_matches, add_inst_num)
from cosypose.utils.xarray import xr_merge


class DetectionMeter(Meter):
    def __init__(self,
                 iou_threshold=0.5,
                 errors_bsz=512,
                 consider_all_predictions=False,
                 targets=None,
                 visib_gt_min=-1,
                 n_top=-1):

        self.iou_threshold = iou_threshold
        self.consider_all_predictions = consider_all_predictions
        self.targets = targets
        self.visib_gt_min = visib_gt_min
        self.errors_bsz = errors_bsz
        self.n_top = n_top
        self.reset()

    def compute_metrics(self, bbox_pred, bbox_gt):
        iou_all = torchvision.ops.box_iou(bbox_pred, bbox_gt)
        arange_n = torch.arange(len(bbox_pred))
        iou = iou_all[arange_n, arange_n]
        return dict(iou=iou)

    def compute_metrics_batch(self, bbox_pred, bbox_gt):
        metrics = []
        ids = torch.arange(len(bbox_pred))
        ds = TensorDataset(bbox_pred, bbox_gt, ids)
        dl = DataLoader(ds, batch_size=self.errors_bsz)

        for (bbox_pred_, bbox_gt_, ids_) in dl:
            metrics.append(self.compute_metrics(bbox_pred_, bbox_gt_))

        if len(metrics) == 0:
            metrics.append(dict(
                iou=torch.empty(0, dtype=torch.float),
            ))

        metricsd = dict()
        for k in metrics[0].keys():
            metricsd[k] = torch.cat([metrics_n[k] for metrics_n in metrics], dim=0)
        return metricsd

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
        pred_ids = cand_infos['pred_id'].values.tolist()
        gt_ids = cand_infos['gt_id'].values.tolist()
        cand_bbox_gt = gt_data.bboxes[gt_ids]
        cand_bbox_pred = pred_data_filtered.bboxes[pred_ids]

        # Compute metrics for tentative matches
        metrics = self.compute_metrics_batch(cand_bbox_pred, cand_bbox_gt)

        # Matches can only be candidates within thresholds
        cand_infos['iou'] = metrics['iou'].cpu().numpy()
        keep = cand_infos['iou'] >= self.iou_threshold
        cand_infos = cand_infos[keep].reset_index(drop=True)

        # Match predictions to ground truth detections
        cand_infos['error'] = - cand_infos['iou']
        matches = match_poses(cand_infos, group_keys=group_keys)

        # Save all informations in xarray datasets
        gt_keys = group_keys + ['gt_inst_id', 'valid'] + (['visib_fract'] if 'visib_fract' in gt_infos else [])
        gt = gt_data.infos.loc[:, gt_keys]
        preds = pred_data.infos.loc[:, group_keys + ['pred_inst_id', 'score']]
        matches = matches.loc[:, group_keys + ['pred_inst_id', 'gt_inst_id', 'cand_id']]

        gt = xr.Dataset(gt).rename({'dim_0': 'gt_id'})
        matches = xr.Dataset(matches).rename({'dim_0': 'match_id'})
        preds = xr.Dataset(preds).rename({'dim_0': 'pred_id'})

        ious = metrics['iou'].cpu().numpy()[matches['cand_id'].values]
        matches['iou'] = 'match_id', ious
        matches['iou_valid'] = 'match_id', ious >= self.iou_threshold

        fill_values = {
            'iou': np.nan,
            'iou_valid': False,
            'score': np.nan,
        }
        matches = xr_merge(matches, preds, on=group_keys + ['pred_inst_id'],
                           dim1='match_id', dim2='pred_id', fill_value=fill_values)
        gt = xr_merge(gt, matches, on=group_keys + ['gt_inst_id'],
                      dim1='gt_id', dim2='match_id', fill_value=fill_values)

        preds_match_merge = xr_merge(preds, matches, on=group_keys+['pred_inst_id'],
                                     dim1='pred_id', dim2='match_id', fill_value=fill_values)
        preds['iou_valid'] = 'pred_id', preds_match_merge['iou_valid']

        self.datas['gt_df'].append(gt)
        self.datas['pred_df'].append(preds)
        self.datas['matches_df'].append(matches)

    def summary(self):
        gt_df = xr.concat(self.datas['gt_df'], dim='gt_id')
        matches_df = xr.concat(self.datas['matches_df'], dim='match_id')
        pred_df = xr.concat(self.datas['pred_df'], dim='pred_id')
        valid_df = gt_df.sel(gt_id=gt_df['valid'])

        # AP/mAP @ IoU < threshold
        valid_k = 'iou_valid'
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
            'iou_valid_recall': valid_df['iou_valid'].sum('gt_id').item() / n_gt_valid,
        }

        summary.update({
            'AP': AP,
            'mAP': mAP,
        })

        dfs = dict(gt=gt_df, matches=matches_df, preds=pred_df, ap=ap_dfs)
        return summary, dfs
