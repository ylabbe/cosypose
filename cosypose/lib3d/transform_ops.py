import transforms3d
import numpy as np
import torch
from .rotations import compute_rotation_matrix_from_ortho6d


def transform_pts(T, pts):
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError('Unsupported shape for T', T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, [-1]]
    R_inv = R.transpose(-2, -1)
    t_inv = - R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, [-1]] = t_inv
    return T_inv


def add_noise(TCO, euler_deg_std=[15, 15, 15], trans_std=[0.01, 0.01, 0.05]):
    TCO_out = TCO.clone()
    device = TCO_out.device
    bsz = TCO.shape[0]
    euler_noise_deg = np.concatenate(
        [np.random.normal(loc=0, scale=euler_deg_std_i, size=bsz)[:, None]
         for euler_deg_std_i in euler_deg_std], axis=1)
    euler_noise_rad = euler_noise_deg * np.pi / 180
    R_noise = torch.tensor([transforms3d.euler.euler2mat(*xyz) for xyz in euler_noise_rad]).float().to(device)

    trans_noise = np.concatenate(
        [np.random.normal(loc=0, scale=trans_std_i, size=bsz)[:, None]
         for trans_std_i in trans_std], axis=1)
    trans_noise = torch.tensor(trans_noise).float().to(device)
    TCO_out[:, :3, :3] = TCO_out[:, :3, :3] @ R_noise
    TCO_out[:, :3, 3] += trans_noise
    return TCO_out


def compute_transform_from_pose9d(pose9d):
    # assert len(pose9d.shape) == 2
    # assert pose9d.shape[1] == 9
    assert pose9d.shape[-1] == 9
    R = compute_rotation_matrix_from_ortho6d(pose9d[..., :6])
    trans = pose9d[..., 6:]
    T = torch.zeros(*pose9d.shape[:-1], 4, 4, dtype=pose9d.dtype, device=pose9d.device)
    T[..., 0:3, 0:3] = R
    T[..., 0:3, 3] = trans
    T[..., 3, 3] = 1
    return T
