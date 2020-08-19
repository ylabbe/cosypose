import torch

from .rotations import compute_rotation_matrix_from_ortho6d, compute_rotation_matrix_from_quaternions
from .transform_ops import transform_pts

l1 = lambda diff: diff.abs()
l2 = lambda diff: diff ** 2


def apply_imagespace_predictions(TCO, K, vxvyvz, dRCO):
    assert TCO.shape[-2:] == (4, 4)
    assert K.shape[-2:] == (3, 3)
    assert dRCO.shape[-2:] == (3, 3)
    assert vxvyvz.shape[-1] == 3
    TCO_out = TCO.clone()

    # Translation in image space
    zsrc = TCO[:, 2, [3]]
    vz = vxvyvz[:, [2]]
    ztgt = vz * zsrc

    vxvy = vxvyvz[:, :2]
    fxfy = K[:, [0, 1], [0, 1]]
    xsrcysrc = TCO[:, :2, 3]
    TCO_out[:, 2, 3] = ztgt.flatten()
    TCO_out[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / zsrc.repeat(1, 2))) * ztgt.repeat(1, 2)

    # Rotation in camera frame
    # TC1' = TC2' @  T2'1' where TC2' = T22' = dCRO is predicted and T2'1'=T21=TC1
    TCO_out[:, :3, :3] = dRCO @ TCO[:, :3, :3]
    return TCO_out


def loss_CO_symmetric(TCO_possible_gt, TCO_pred, points, l1_or_l2=l1):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert len(TCO_possible_gt.shape) == 4 and TCO_possible_gt.shape[-2:] == (4, 4)
    assert TCO_pred.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3

    TCO_points_possible_gt = transform_pts(TCO_possible_gt, points)
    TCO_pred_points = transform_pts(TCO_pred, points)
    losses_possible = l1_or_l2((TCO_pred_points.unsqueeze(1) - TCO_points_possible_gt).flatten(-2, -1)).mean(-1)
    loss, min_id = losses_possible.min(dim=1)
    TCO_assign = TCO_possible_gt[torch.arange(bsz), min_id]
    return loss, TCO_assign


def loss_refiner_CO_disentangled(TCO_possible_gt,
                                 TCO_input, refiner_outputs,
                                 K_crop, points):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert TCO_input.shape[0] == bsz
    assert refiner_outputs.shape == (bsz, 9)
    assert K_crop.shape == (bsz, 3, 3)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3
    assert TCO_possible_gt.dim() == 4 and TCO_possible_gt.shape[-2:] == (4, 4)

    dR = compute_rotation_matrix_from_ortho6d(refiner_outputs[:, 0:6])
    vxvyvz = refiner_outputs[:, 6:9]
    TCO_gt = TCO_possible_gt[:, 0]

    TCO_pred_orn = TCO_gt.clone()
    TCO_pred_orn[:, :3, :3] = dR @ TCO_input[:, :3, :3]

    TCO_pred_xy = TCO_gt.clone()
    z_gt = TCO_gt[:, 2, [3]]
    z_input = TCO_input[:, 2, [3]]
    vxvy = vxvyvz[:, :2]
    fxfy = K_crop[:, [0, 1], [0, 1]]
    xsrcysrc = TCO_input[:, :2, 3]
    TCO_pred_xy[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / z_input.repeat(1, 2))) * z_gt.repeat(1, 2)

    TCO_pred_z = TCO_gt.clone()
    vz = vxvyvz[:, [2]]
    TCO_pred_z[:, [2], [3]] = vz * z_input

    loss_orn, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_orn, points, l1_or_l2=l1)
    loss_xy, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_xy, points, l1_or_l2=l1)
    loss_z, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_z, points, l1_or_l2=l1)
    return loss_orn + loss_xy + loss_z


def loss_refiner_CO_disentangled_quaternions(TCO_possible_gt,
                                             TCO_input, refiner_outputs,
                                             K_crop, points):
    bsz = TCO_possible_gt.shape[0]
    assert TCO_possible_gt.shape[0] == bsz
    assert TCO_input.shape[0] == bsz
    assert refiner_outputs.shape == (bsz, 7)
    assert K_crop.shape == (bsz, 3, 3)
    assert points.dim() == 3 and points.shape[0] == bsz and points.shape[-1] == 3
    assert TCO_possible_gt.dim() == 4 and TCO_possible_gt.shape[-2:] == (4, 4)

    dR = compute_rotation_matrix_from_quaternions(refiner_outputs[:, 0:4])
    vxvyvz = refiner_outputs[:, 4:7]
    TCO_gt = TCO_possible_gt[:, 0]

    TCO_pred_orn = TCO_gt.clone()
    TCO_pred_orn[:, :3, :3] = dR @ TCO_input[:, :3, :3]

    TCO_pred_xy = TCO_gt.clone()
    z_gt = TCO_gt[:, 2, [3]]
    z_input = TCO_input[:, 2, [3]]
    vxvy = vxvyvz[:, :2]
    fxfy = K_crop[:, [0, 1], [0, 1]]
    xsrcysrc = TCO_input[:, :2, 3]
    TCO_pred_xy[:, :2, 3] = ((vxvy / fxfy) + (xsrcysrc / z_input.repeat(1, 2))) * z_gt.repeat(1, 2)

    TCO_pred_z = TCO_gt.clone()
    vz = vxvyvz[:, [2]]
    TCO_pred_z[:, [2], [3]] = vz * z_input

    loss_orn, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_orn, points, l1_or_l2=l1)
    loss_xy, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_xy, points, l1_or_l2=l1)
    loss_z, _ = loss_CO_symmetric(TCO_possible_gt, TCO_pred_z, points, l1_or_l2=l1)
    return loss_orn + loss_xy + loss_z


def TCO_init_from_boxes(z_range, boxes, K):
    # Used in the paper
    assert len(z_range) == 2
    assert boxes.shape[-1] == 4
    assert boxes.dim() == 2
    bsz = boxes.shape[0]
    uv_centers = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2
    z = torch.as_tensor(z_range).mean().unsqueeze(0).unsqueeze(0).repeat(bsz, 1).to(boxes.device).to(boxes.dtype)
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    xy_init = ((uv_centers - cxcy) * z) / fxfy
    TCO = torch.eye(4).unsqueeze(0).to(torch.float).to(boxes.device).repeat(bsz, 1, 1)
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


def TCO_init_from_boxes_zup_autodepth(boxes_2d, model_points_3d, K):
    # User in BOP20 challenge
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, z_guess],
        [0, 0, 0, 1]
    ]).to(torch.float).to(boxes_2d.device).repeat(bsz, 1, 1)
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)
    deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
    deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay

    # z = z_from_dx.unsqueeze(1)
    # z = z_from_dy.unsqueeze(1)
    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO
