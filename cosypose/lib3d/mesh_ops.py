import torch
import numpy as np


def get_meshes_center(pts):
    bsz = pts.shape[0]
    limits = get_meshes_bounding_boxes(pts)
    t_offset = limits[..., :3].mean(dim=1)
    T_offset = torch.eye(4, 4, dtype=pts.dtype, device=pts.device)
    T_offset = T_offset.unsqueeze(0).repeat(bsz, 1, 1)
    T_offset[:, :3, -1] = t_offset
    return T_offset


def get_meshes_bounding_boxes(pts):
    xmin, xmax = pts[..., 0].min(dim=-1, keepdim=True).values, pts[..., 0].max(dim=-1, keepdim=True).values
    ymin, ymax = pts[..., 1].min(dim=-1, keepdim=True).values, pts[..., 1].max(dim=-1, keepdim=True).values
    zmin, zmax = pts[..., 2].min(dim=-1, keepdim=True).values, pts[..., 2].max(dim=-1, keepdim=True).values
    v0 = torch.cat((xmin, ymax, zmax), dim=-1).unsqueeze(1)
    v1 = torch.cat((xmax, ymax, zmax), dim=-1).unsqueeze(1)
    v2 = torch.cat((xmax, ymin, zmax), dim=-1).unsqueeze(1)
    v3 = torch.cat((xmin, ymin, zmax), dim=-1).unsqueeze(1)
    v4 = torch.cat((xmin, ymax, zmin), dim=-1).unsqueeze(1)
    v5 = torch.cat((xmax, ymax, zmin), dim=-1).unsqueeze(1)
    v6 = torch.cat((xmax, ymin, zmin), dim=-1).unsqueeze(1)
    v7 = torch.cat((xmin, ymin, zmin), dim=-1).unsqueeze(1)
    bbox_pts = torch.cat((v0, v1, v2, v3, v4, v5, v6, v7), dim=1)
    return bbox_pts


def sample_points(points, n_points, deterministic=False):
    assert points.dim() == 3
    assert n_points <= points.shape[1]
    if deterministic:
        np_random = np.random.RandomState(0)
    else:
        np_random = np.random
    point_ids = np_random.choice(points.shape[1], size=n_points, replace=False)
    point_ids = torch.as_tensor(point_ids).to(points.device)
    points = torch.index_select(points, 1, point_ids)
    return points
