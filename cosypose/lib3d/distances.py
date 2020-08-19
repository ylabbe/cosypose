import torch
from cosypose.lib3d.transform_ops import transform_pts


def dists_add(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points - TXO_pred_points
    return dists


def dists_add_symmetric(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)
    dists_norm_squared = (dists ** 2).sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    dists = dists[ids_row, assign, ids_col]
    return dists
