import numpy as np
import time
import transforms3d
import torch
from copy import deepcopy
from collections import defaultdict
import seaborn as sns

from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.lib3d.transform import Transform
from cosypose.lib3d.transform_ops import invert_T
from cosypose.lib3d.rotations import euler2quat
from .plotter import Plotter


def get_group_infos(group_id, mv_scene_ds):
    mask = mv_scene_ds.frame_index['group_id'] == group_id
    row = mv_scene_ds.frame_index.loc[mask]
    scene_id, view_ids = row.scene_id.item(), row.view_ids.item()
    return scene_id, view_ids


def filter_predictions(preds, group_id):
    m = preds.infos['group_id'] == group_id
    return preds[np.where(m)[0]]


def nms3d(preds, th=0.04, poses_attr='poses'):
    TCO, TCO_infos = getattr(preds, poses_attr).cpu(), preds.infos
    is_tested = set()
    TCO = np.array(TCO)
    scores = TCO_infos['score'].values
    all_t = TCO[:, :3, -1]
    argsort = np.argsort(-scores)
    keep = []
    for idx, TCO_n in zip(argsort, TCO[argsort]):
        if idx in is_tested:
            continue
        t_n = TCO_n[:3, -1]
        dists = np.linalg.norm(t_n - all_t, axis=-1)
        dists[idx] = np.inf
        ids_merge = np.where(dists <= th)[0]
        for id_merge in ids_merge:
            is_tested.add(id_merge)
        keep.append(idx)
    TCO = TCO[keep]
    TCO_infos = TCO_infos.loc[keep].reset_index(drop=True)

    new_preds = preds.clone()
    new_preds.infos = TCO_infos
    new_preds.poses = torch.as_tensor(TCO)
    return new_preds


def make_scene_renderings(objects, cameras, urdf_ds_name, distance=1.5, theta=np.pi/4, angles=[0],
                          object_scale=1.0, camera_scale=1.5, background_color=(242, 231, 191),
                          show_cameras=False,
                          resolution=(640, 480), colormap_rgb=None, object_id_ref=0,
                          gui=False,
                          use_nms3d=True,
                          camera_color=(0.2, 0.2, 0.2, 1.0)):

    renderer = BulletSceneRenderer([urdf_ds_name, 'camera'], background_color=background_color, gui=gui)
    urdf_ds = renderer.body_cache.urdf_ds

    # Patch the scales for visualization
    is_camera = np.array(['camera' in label for label in urdf_ds.index['label']])
    urdf_ds.index.loc[~is_camera, 'scale'] = object_scale * 0.001
    urdf_ds.index.loc[is_camera, 'scale'] = camera_scale

    if use_nms3d:
        objects = nms3d(objects, poses_attr='TWO', th=0.04)
    objects = objects.cpu()
    objects.TWO = objects.poses

    if colormap_rgb is None:
        colormap_rgb, _ = make_colormaps(objects.infos['label'])
    objects.infos['color'] = objects.infos['label'].apply(lambda k: colormap_rgb[k])

    cameras = cameras.cpu()
    TWWB = objects.poses[object_id_ref]

    cam = cameras[[0]]
    TCWB = invert_T(cam.TWC.squeeze(0)) @ TWWB
    TWBC = invert_T(TCWB)
    if TWBC[2, -1] < 0:
        quat = euler2quat([np.pi, 0, 0])
        TWWB = Transform(TWWB.numpy()) * Transform(quat, np.zeros(3))
        TWWB = TWWB.toHomogeneousMatrix()
    TWWB = np.asarray(TWWB)

    list_objects = []
    for obj_id in range(len(objects)):
        TWO = np.linalg.inv(TWWB) @ objects.TWO[obj_id].numpy()
        TWO[:3, -1] *= object_scale
        obj = dict(
            name=objects.infos.loc[obj_id, 'label'],
            color=objects.infos.loc[obj_id, 'color'],
            TWO=TWO,
        )
        list_objects.append(obj)
    target = np.mean(np.stack([obj['TWO'][:3, -1] for obj in list_objects]), axis=0)

    if show_cameras:
        for cam_id in range(len(cameras)):
            obj = dict(
                name='camera',
                color=camera_color,
                TWO=np.linalg.inv(TWWB) @ cameras.TWC[cam_id].numpy()
            )
            list_objects.append(obj)

    fx, fy = 515, 515
    w, h = resolution
    K = np.array([
        [fx, 0, w/2],
        [0, fy, h/2],
        [0, 0, 1]
    ])
    list_cameras = []
    for phi in angles:
        x = distance * np.sin(theta) * np.cos(phi)
        y = distance * np.sin(theta) * np.sin(phi)
        z = distance * np.cos(theta)
        t = np.array([x, y, z])
        R = transforms3d.euler.euler2mat(np.pi, theta, phi, axes='sxyz')
        R = R @ transforms3d.euler.euler2mat(0, 0, -np.pi/2, axes='sxyz')
        t += np.array(target)
        TWC = Transform(R, t).toHomogeneousMatrix()
        TWBC = TWWB @ TWC
        list_cameras.append(
            dict(K=K, TWC=TWC, resolution=(w, h))
        )
    renders = renderer.render_scene(list_objects, list_cameras)
    images = np.stack([render['rgb'] for render in renders])
    if gui:
        time.sleep(100)
    renderer.disconnect()
    return images


def make_colormaps(labels):
    colors_hex = sns.color_palette(n_colors=len(labels)).as_hex()
    colormap_hex = {label: color for label, color in zip(labels, colors_hex)}
    colormap_rgb = {k: [int(h[1:][i:i+2], 16) / 255. for i in (0, 2, 4)] + [1.0] for k, h in colormap_hex.items()}
    return colormap_rgb, colormap_hex


def mark_inliers(cand_inputs, cand_matched):
    inliers_infos = cand_matched.infos[['scene_id', 'view_id', 'label', 'cand_id']].copy()
    inliers_infos['is_inlier'] = True
    infos = cand_inputs.infos.merge(inliers_infos, on=['scene_id', 'view_id', 'label', 'cand_id'], how='left')
    infos['is_inlier'] = infos['is_inlier'].astype(np.float)
    infos.loc[~np.isfinite(infos.loc[:, 'is_inlier'].astype(np.float)), 'is_inlier'] = 0
    infos['is_inlier'] = infos['is_inlier'].astype(np.bool)
    cand_inputs.infos = infos
    return cand_inputs


def render_predictions_wrt_camera(renderer, preds_with_colors, camera):
    for k in ('K', 'resolution'):
        assert k in camera

    camera = deepcopy(camera)
    camera.update(TWC=np.eye(4))
    list_objects = []
    for n in range(len(preds_with_colors)):
        row = preds_with_colors.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=row.color,
            TWO=preds_with_colors.poses[n].cpu().numpy(),
        )
        list_objects.append(obj)
    rgb_rendered = renderer.render_scene(list_objects, [camera])[0]['rgb']
    return rgb_rendered


def render_gt(renderer, objects, camera, colormap_rgb):
    camera = deepcopy(camera)
    TWC = camera['TWC']
    for obj in objects:
        obj['color'] = colormap_rgb[obj['label']]
        obj['TWO'] = np.linalg.inv(TWC) @ obj['TWO']
    camera['TWC'] = np.eye(4)
    rgb_rendered = renderer.render_scene(objects, [camera])[0]['rgb']
    return rgb_rendered


def add_colors_to_predictions(predictions, colormap):
    predictions.infos['color'] = predictions.infos['label'].map(colormap)
    return predictions


def make_cosypose_plots(scene_ds, scene_id, view_ids,
                        dict_predictions, renderer,
                        use_class_colors_for_3d=True,
                        use_nms3d=True,
                        inlier_color=(0, 1, 0, 1.0),
                        outlier_color=(1.0, 0, 0, 0.3)):
    plotter = Plotter()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])

    _, _, gt_state = scene_ds[scene_ds_index.loc[(scene_id, view_ids[0]), 'ds_idx']]
    scene_labels = set([obj['label'] for obj in gt_state['objects']])

    preds_by_view = dict()
    for view_id in view_ids:
        this_view_dict_preds = dict()
        for k in ('cand_inputs', 'cand_matched', 'ba_output'):
            assert k in dict_predictions
            scene_labels = scene_labels.union(set(dict_predictions[k].infos['label'].values.tolist()))
            pred_infos = dict_predictions[k].infos
            keep = np.logical_and(pred_infos['scene_id'] == scene_id,
                                  np.isin(pred_infos['view_id'], view_id))
            this_view_dict_preds[k] = dict_predictions[k][np.where(keep)[0]]
        preds_by_view[view_id] = this_view_dict_preds

    colormap_rgb, colormap_hex = make_colormaps(scene_labels)
    colormap_rgb_3d = colormap_rgb if use_class_colors_for_3d else defaultdict(lambda: (1, 1, 1, 1))

    fig_array = []
    for view_id in view_ids:
        this_view_dict_preds = preds_by_view[view_id]
        input_rgb, _, gt_state = scene_ds[scene_ds_index.loc[(scene_id, view_id), 'ds_idx']]
        fig_input_im = plotter.plot_image(input_rgb)

        # Detections
        detections = this_view_dict_preds['cand_inputs']
        bboxes = detections.initial_bboxes
        bboxes = bboxes + torch.as_tensor(np.random.randint(30, size=((len(bboxes), 4)))).to(bboxes.dtype).to(bboxes.device)
        detections.bboxes = bboxes

        detections = add_colors_to_predictions(detections, colormap_hex)
        fig_detections = plotter.plot_image(input_rgb)
        fig_detections = plotter.plot_maskrcnn_bboxes(fig_detections, detections, colors=detections.infos['color'].tolist())
        # fig_array.append([fig_input_im, fig_detections])

        # Candidates
        cand_inputs = this_view_dict_preds['cand_inputs']
        cand_matched = this_view_dict_preds['cand_matched']
        cand_inputs = mark_inliers(cand_inputs, cand_matched)
        colors = np.array([inlier_color if is_inlier else outlier_color for is_inlier in cand_inputs.infos['is_inlier']])
        cand_inputs.infos['color'] = colors.tolist()

        cand_rgb_rendered = render_predictions_wrt_camera(renderer, cand_inputs, gt_state['camera'])
        fig_cand = plotter.plot_overlay(input_rgb, cand_rgb_rendered)

        # Scene reconstruction
        ba_outputs = this_view_dict_preds['ba_output']
        if use_nms3d:
            ba_outputs = nms3d(ba_outputs)
        ba_outputs = add_colors_to_predictions(ba_outputs, colormap_rgb_3d)

        outputs_rgb_rendered = render_predictions_wrt_camera(renderer, ba_outputs, gt_state['camera'])
        fig_outputs = plotter.plot_overlay(input_rgb, outputs_rgb_rendered)

        gt_rgb_rendered = render_gt(renderer, gt_state['objects'], gt_state['camera'], colormap_rgb_3d)
        fig_gt = plotter.plot_overlay(input_rgb, gt_rgb_rendered)

        fig_array.append([fig_input_im, fig_detections, fig_cand, fig_outputs, fig_gt])
    return fig_array


def render_candidates(scene_ds, scene_id, view_ids,
                      colormap_rgb, dict_predictions, renderer):

    plotter = Plotter()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])

    preds_by_view = dict()
    for view_id in view_ids:
        this_view_dict_preds = dict()
        for k in ('cand_inputs', 'cand_matched', 'ba_output'):
            assert k in dict_predictions
            pred_infos = dict_predictions[k].infos
            keep = np.logical_and(pred_infos['scene_id'] == scene_id,
                                  np.isin(pred_infos['view_id'], view_id))
            this_view_dict_preds[k] = dict_predictions[k][np.where(keep)[0]]
        preds_by_view[view_id] = this_view_dict_preds

    figures_by_view = []
    for view_id in view_ids:
        this_view_figures = dict()
        figures_by_view.append(this_view_figures)

        this_view_dict_preds = preds_by_view[view_id]
        input_rgb, _, gt_state = scene_ds[scene_ds_index.loc[(scene_id, view_id), 'ds_idx']]
        fig_input_im = plotter.plot_image(input_rgb)
        this_view_figures['input_im'] = fig_input_im

        # Detections
        detections = this_view_dict_preds['cand_inputs']
        bboxes = detections.initial_bboxes
        bboxes = bboxes + torch.as_tensor(np.random.randint(10, size=((len(bboxes), 4)))).to(bboxes.dtype).to(bboxes.device)
        detections.bboxes = bboxes
        this_view_figures['input_im'] = fig_input_im

        detections = add_colors_to_predictions(detections, lambda k: '#FFFF00')
        fig_detections = plotter.plot_image(input_rgb)
        fig_detections = plotter.plot_maskrcnn_bboxes(fig_detections, detections,
                                                      text_auto=False,
                                                      colors=detections.infos['color'].tolist())
        this_view_figures['detections'] = fig_detections

        detections = add_colors_to_predictions(detections, colormap_rgb)
        fig_candidates = []
        fig_candidates_black = []
        for cand_id in range(len(detections)):
            cand_pred = detections[[cand_id]]
            rgb_rendered = render_predictions_wrt_camera(renderer, cand_pred, gt_state['camera'])
            fig = plotter.plot_overlay(input_rgb, rgb_rendered)
            fig_candidates.append(fig)
            fig = plotter.plot_image(rgb_rendered)
            fig_candidates_black.append(fig)
        this_view_figures['candidates'] = fig_candidates
        this_view_figures['candidates_black'] = fig_candidates_black
    return figures_by_view, preds_by_view
