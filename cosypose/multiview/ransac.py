import torch
from collections import defaultdict
import pandas as pd
import numpy as np
import cosypose_cext
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import cosypose.utils.tensor_collection as tc

from cosypose.lib3d.transform_ops import invert_T
from cosypose.lib3d.symmetric_distances import (
    symmetric_distance_batched_fast, expand_ids_for_symmetry, scatter_argmin)

from cosypose.utils.logging import get_logger
from cosypose.utils.timer import Timer
logger = get_logger(__name__)


def estimate_camera_poses(TC1Oa, TC2Ob, labels_ab,
                          TC1Og, TC2Od, labels_gd,
                          mesh_db):
    # Assume (TC1Oa and TC2Ob), (TC1Og, TC2Od) are the same.
    # Notation differ from the paper, paper(code)
    # we have 1(a), 2(b), a(alpha), b(beta), g(gamma), d(delta)
    bsz = TC1Oa.shape[0]
    assert TC1Oa.shape == (bsz, 4, 4)
    assert TC2Ob.shape == (bsz, 4, 4)
    assert TC1Og.shape == (bsz, 4, 4)
    assert TC2Od.shape == (bsz, 4, 4)
    assert len(labels_ab) == bsz
    assert len(labels_gd) == bsz
    TObC2 = invert_T(TC2Ob)

    meshes_ab = mesh_db.select(labels_ab)
    ids_expand, sym_ids = expand_ids_for_symmetry(labels_ab, mesh_db.n_sym_mapping)
    sym_expand = meshes_ab.symmetries[ids_expand, sym_ids]

    dist_fn = symmetric_distance_batched_fast
    dists, _ = dist_fn(
        TC1Og[ids_expand],
        (TC1Oa[ids_expand] @ sym_expand @ TObC2[ids_expand]) @ TC2Od[ids_expand],
        labels_gd[ids_expand], mesh_db
    )
    min_ids = scatter_argmin(dists, ids_expand)
    S_Oa_star = meshes_ab.symmetries[torch.arange(len(min_ids)), sym_ids[min_ids]]
    TC1C2 = TC1Oa @ S_Oa_star @ TObC2
    return TC1C2


def estimate_camera_poses_batch(candidates, seeds, mesh_db, bsz=1024):
    n_tot = len(seeds['match1_cand1'])
    n_batch = max(1, int(n_tot // bsz))
    ids_split = np.array_split(np.arange(n_tot), n_batch)
    all_TC1C2 = []
    for ids in ids_split:
        labels_ab = candidates.infos['label'].iloc[seeds['match1_cand1'][ids]].values
        labels_gd = candidates.infos['label'].iloc[seeds['match2_cand1'][ids]].values
        TC1Oa = candidates.poses[seeds['match1_cand1'][ids]]
        TC2Ob = candidates.poses[seeds['match1_cand2'][ids]]
        TC1Og = candidates.poses[seeds['match2_cand1'][ids]]
        TC2Od = candidates.poses[seeds['match2_cand2'][ids]]
        TC1C2 = estimate_camera_poses(TC1Oa, TC2Ob, labels_ab, TC1Og, TC2Od, labels_gd, mesh_db)
        all_TC1C2.append(TC1C2)
    return torch.cat(all_TC1C2, dim=0)


def score_tmatches(TC1Oa, TC2Ob, TC1C2, labels_ab, mesh_db):
    TWOa = TC1Oa
    TWOb = TC1C2 @ TC2Ob

    dist_fn = symmetric_distance_batched_fast
    dists, _ = dist_fn(TWOa, TWOb, labels_ab, mesh_db)
    return dists


def score_tmaches_batch(candidates, tmatches, TC1C2, mesh_db, bsz=4096):
    n_tot = len(tmatches['cand1'])
    n_batch = max(1, int(n_tot // bsz))
    ids_split = np.array_split(np.arange(n_tot), n_batch)
    all_dists = []
    for ids in ids_split:
        labels = candidates.infos['label'].iloc[tmatches['cand1'][ids]].values
        TC1Oa = candidates.poses[tmatches['cand1'][ids]]
        TC2Ob = candidates.poses[tmatches['cand2'][ids]]
        TC1C2_ = TC1C2[tmatches['hypothesis_id'][ids]]
        dists = score_tmatches(TC1Oa, TC2Ob, TC1C2_, labels, mesh_db)
        all_dists.append(dists)
    return torch.cat(all_dists, dim=0)


def scene_level_matching(candidates, inliers):
    cand1 = inliers['inlier_matches_cand1']
    cand2 = inliers['inlier_matches_cand2']
    edges = np.ones((len(cand1)), dtype=np.int)
    n_cand = len(candidates)
    graph = csr_matrix((edges, (cand1, cand2)), shape=(n_cand, n_cand))
    n_components, ids = connected_components(graph, directed=True, connection='strong')

    component_size = defaultdict(lambda: 0)
    for idx in ids:
        component_size[idx] += 1
    obj_n_cand = np.empty(len(ids), dtype=np.int)
    for n, idx in enumerate(ids):
        obj_n_cand[n] = component_size[idx]

    cand_infos = candidates.infos.copy()
    cand_infos['component_id'] = ids
    keep_cand = obj_n_cand >= 2
    cand_infos = cand_infos[keep_cand].reset_index(drop=True)
    for n, (comp_id, group) in enumerate(cand_infos.groupby('component_id')):
        cand_infos.loc[group.index, 'component_id'] = n
    cand_infos = cand_infos.rename(columns={'component_id': 'obj_id'})

    matched_candidates = tc.PandasTensorCollection(infos=cand_infos,
                                                   poses=candidates.poses[cand_infos['cand_id'].values])
    return matched_candidates


def make_obj_infos(matched_candidates):
    scene_infos = matched_candidates.infos.loc[:, ['obj_id', 'score', 'label']].copy()
    gb = scene_infos.groupby('obj_id')
    scene_infos['n_cand'] = gb['score'].transform(len).astype(np.int)
    scene_infos['score'] = gb['score'].transform(np.sum)
    scene_infos = gb.first().reset_index(drop=False)
    return scene_infos


def get_best_viewpair_pose_est(TC1C2, seeds, inliers):
    best_hypotheses = inliers['best_hypotheses']
    TC1C2_best = TC1C2[best_hypotheses]
    view1 = seeds['view1'][best_hypotheses]
    view2 = seeds['view2'][best_hypotheses]
    infos = pd.DataFrame(dict(view1=view1, view2=view2))
    return tc.PandasTensorCollection(infos=infos, TC1C2=TC1C2_best)


def multiview_candidate_matching(candidates, mesh_db,
                                 model_bsz=1e3,
                                 score_bsz=1e5,
                                 dist_threshold=0.02,
                                 cameras=None,
                                 n_ransac_iter=20,
                                 n_min_inliers=3):
    timer_models = Timer()
    timer_score = Timer()
    timer_misc = Timer()

    known_poses = cameras is not None
    if known_poses:
        logger.debug('Using known camera poses.')
        n_ransac_iter = 1
    else:
        logger.debug('Estimating camera poses using RANSAC.')

    timer_misc.start()
    candidates.infos['cand_id'] = np.arange(len(candidates))
    timer_misc.pause()

    timer_models.start()
    seeds, tmatches = cosypose_cext.make_ransac_infos(
        candidates.infos['view_id'].values.tolist(), candidates.infos['label'].values.tolist(),
        n_ransac_iter, 0,
    )

    if not known_poses:
        TC1C2 = estimate_camera_poses_batch(candidates, seeds, mesh_db, bsz=model_bsz)
    else:
        cameras.infos['idx'] = np.arange(len(cameras))
        view_map = cameras.infos.set_index('view_id')
        TWC1 = cameras.TWC[view_map.loc[seeds['view1'], 'idx'].values]
        TWC2 = cameras.TWC[view_map.loc[seeds['view2'], 'idx'].values]
        TC1C2 = invert_T(TWC1) @ TWC2
    timer_models.pause()

    timer_score.start()
    dists = score_tmaches_batch(candidates, tmatches, TC1C2, mesh_db, bsz=score_bsz)

    inliers = cosypose_cext.find_ransac_inliers(
        seeds['view1'], seeds['view2'],
        tmatches['hypothesis_id'], tmatches['cand1'], tmatches['cand2'],
        dists.cpu().numpy(), dist_threshold, n_min_inliers,
    )
    timer_score.pause()

    timer_misc.start()
    pairs_TC1C2 = get_best_viewpair_pose_est(TC1C2, seeds, inliers)
    filtered_candidates = scene_level_matching(candidates, inliers)
    scene_infos = make_obj_infos(filtered_candidates)
    timer_misc.pause()

    outputs = dict(
        filtered_candidates=filtered_candidates,
        scene_infos=scene_infos,
        pairs_TC1C2=pairs_TC1C2,
        time_models=timer_models.stop(),
        time_score=timer_score.stop(),
        time_misc=timer_misc.stop(),
    )
    return outputs
