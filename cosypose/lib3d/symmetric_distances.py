import torch
import cosypose_cext

from .transform_ops import transform_pts
from .camera_geometry import project_points


def expand_ids_for_symmetry(labels, n_symmetries):
    ids_expand, sym_ids = cosypose_cext.expand_ids_for_symmetry(labels, n_symmetries)
    return ids_expand, sym_ids


def scatter_argmin(dists, ids_expand):
    dists = dists.float().cpu().numpy()
    min_ids = cosypose_cext.scatter_argmin(dists, ids_expand)
    return min_ids


def symmetric_distance_batched(T1, T2, labels, mesh_db):
    bsz = T1.shape[0]
    assert T1.shape == (bsz, 4, 4)
    assert T2.shape == (bsz, 4, 4)
    assert len(labels) == bsz

    meshes = mesh_db.select(labels)
    ids_expand, sym_ids = expand_ids_for_symmetry(labels, mesh_db.n_sym_mapping)
    points_expand = meshes.points[ids_expand]
    sym_expand = meshes.symmetries[ids_expand, sym_ids]
    dists = mesh_points_dist(T1[ids_expand] @ sym_expand,
                             T2[ids_expand],
                             points_expand)
    min_ids = scatter_argmin(dists, ids_expand)
    min_dists = dists[min_ids]
    S12 = meshes.symmetries[torch.arange(len(min_ids)), sym_ids[min_ids]]
    return min_dists, S12


def symmetric_distance_batched_fast(T1, T2, labels, mesh_db):
    bsz = T1.shape[0]
    assert T1.shape == (bsz, 4, 4)
    assert T2.shape == (bsz, 4, 4)
    assert len(labels) == bsz
    if bsz == 0:
        return torch.empty(0, dtype=T1.dtype, device=T1.device), None

    meshes = mesh_db.select(labels)
    points = meshes.points

    T1_points = transform_pts(T1.unsqueeze(1) @ meshes.symmetries, points)
    T2_points = transform_pts(T2, points).unsqueeze(1)

    dists_squared = ((T1_points - T2_points) ** 2).sum(dim=-1)
    best_sym_id = dists_squared.mean(dim=-1).argmin(dim=1)

    min_dists = torch.sqrt(dists_squared[torch.arange(bsz), best_sym_id]).mean(dim=-1)
    S12 = meshes.symmetries[torch.arange(bsz), best_sym_id]
    return min_dists, S12


def chamfer_dist(T1, T2, labels, mesh_db):
    bsz = T1.shape[0]
    assert T1.shape == (bsz, 4, 4)
    assert T2.shape == (bsz, 4, 4)
    assert len(labels) == bsz
    if bsz == 0:
        return torch.empty(0, dtype=T1.dtype, device=T1.device), None

    meshes = mesh_db.select(labels)
    points = meshes.points

    T1_points = transform_pts(T1, points)
    T2_points = transform_pts(T2, points)

    dists = (T1_points.unsqueeze(1) - T2_points.unsqueeze(2)) ** 2

    assign = dists.sum(dim=-1).argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    dists = torch.sqrt(dists[ids_row, assign, ids_col].sum(dim=-1)).mean(dim=-1)
    return dists, None


def mesh_points_dist(T1, T2, points):
    bsz = T1.shape[0]
    assert T1.shape == (bsz, 4, 4)
    assert T2.shape == (bsz, 4, 4)
    n_pts = points.shape[1]
    assert points.shape == (bsz, n_pts, 3)
    T1_pts = transform_pts(T1, points)
    T2_pts = transform_pts(T2, points)
    return torch.norm(T1_pts - T2_pts, dim=-1, p=2).mean(dim=-1)


def reprojected_dist(T1, T2, K, points):
    bsz = T1.shape[0]
    assert T1.shape == (bsz, 4, 4)
    assert T2.shape == (bsz, 4, 4)
    n_pts = points.shape[1]
    assert points.shape == (bsz, n_pts, 3)
    T1_pts = project_points(points, K, T1)
    T2_pts = project_points(points, K, T2)
    return torch.norm(T1_pts - T2_pts, dim=-1, p=2).mean(dim=-1)


def symmetric_distance_reprojected(T1, T2, K, labels, mesh_db):
    bsz = T1.shape[0]
    assert T1.shape == (bsz, 4, 4)
    assert T2.shape == (bsz, 4, 4)
    assert len(labels) == bsz

    meshes = mesh_db.select(labels)
    ids_expand, sym_ids = expand_ids_for_symmetry(labels, mesh_db.n_sym_mapping)
    points_expand = meshes.points[ids_expand]
    sym_expand = meshes.symmetries[ids_expand, sym_ids]
    dists = reprojected_dist(T1[ids_expand] @ sym_expand,
                             T2[ids_expand], K[ids_expand],
                             points_expand)
    min_ids = scatter_argmin(dists, ids_expand)
    min_dists = dists[min_ids]
    S12 = meshes.symmetries[torch.arange(len(min_ids)), sym_ids[min_ids]]
    return min_dists, S12
