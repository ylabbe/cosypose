import numpy as np
from collections import defaultdict
import torch
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import cosypose.utils.tensor_collection as tc

from cosypose.lib3d.transform_ops import invert_T, compute_transform_from_pose9d
from cosypose.lib3d.camera_geometry import project_points
from cosypose.lib3d.symmetric_distances import symmetric_distance_reprojected

from .ransac import make_obj_infos


from cosypose.utils.logging import get_logger
from cosypose.utils.timer import Timer
logger = get_logger(__name__)


def make_view_groups(pairs_TC1C2):
    views = pairs_TC1C2.infos.loc[:, ['view1', 'view2']].values.T
    views = np.unique(views.reshape(-1))
    view_df = pd.DataFrame(dict(view_id=views, view_local_id=np.arange(len(views))))
    view_to_id = view_df.set_index('view_id')
    view1 = view_to_id.loc[pairs_TC1C2.infos.loc[:, 'view1'], 'view_local_id'].values
    view2 = view_to_id.loc[pairs_TC1C2.infos.loc[:, 'view2'], 'view_local_id'].values
    data = np.ones(len(view1))
    n_views = len(views)
    graph = csr_matrix((data, (view1, view2)), shape=(n_views, n_views))
    n_components, ids = connected_components(graph, directed=True, connection='strong')
    view_df['view_group'] = ids
    view_df = view_df.drop(columns=['view_local_id'])
    return view_df


class SamplerError(Exception):
    pass


class MultiviewRefinement:
    def __init__(self, candidates, cameras, pairs_TC1C2, mesh_db):

        self.device, self.dtype = candidates.device, candidates.poses.dtype
        self.mesh_db = mesh_db
        cameras = cameras.to(self.device).to(self.dtype)
        pairs_TC1C2 = pairs_TC1C2.to(self.device).to(self.dtype)

        view_ids = np.unique(candidates.infos['view_id'])
        keep_ids = np.logical_and(
            np.isin(pairs_TC1C2.infos['view1'], view_ids),
            np.isin(pairs_TC1C2.infos['view2'], view_ids),
        )
        pairs_TC1C2 = pairs_TC1C2[np.where(keep_ids)[0]]

        keep_ids = np.where(np.isin(cameras.infos['view_id'], view_ids))[0]
        cameras = cameras[keep_ids]

        self.cam_infos = cameras.infos
        self.view_to_id = {view_id: n for n, view_id in enumerate(self.cam_infos['view_id'])}
        self.K = cameras.K
        self.n_views = len(self.cam_infos)

        self.obj_infos = make_obj_infos(candidates)
        self.obj_to_id = {obj_id: n for n, obj_id in enumerate(self.obj_infos['obj_id'])}
        self.obj_points = self.mesh_db.select(self.obj_infos['label'].values).points
        self.n_points = self.obj_points.shape[1]
        self.n_objects = len(self.obj_infos)

        self.cand = candidates
        self.cand_TCO = candidates.poses
        self.cand_labels = candidates.infos['label']
        self.cand_view_ids = [self.view_to_id[view_id] for view_id in candidates.infos['view_id']]
        self.cand_obj_ids = [self.obj_to_id[obj_id] for obj_id in candidates.infos['obj_id']]
        self.n_candidates = len(self.cand_TCO)
        self.visibility_matrix = self.make_visibility_matrix(self.cand_view_ids, self.cand_obj_ids)

        self.v2v1_TC2C1_map = {(self.view_to_id[v2], self.view_to_id[v1]): invert_T(TC1C2) for
                               (v1, v2, TC1C2) in zip(pairs_TC1C2.infos['view1'],
                                                      pairs_TC1C2.infos['view2'],
                                                      pairs_TC1C2.TC1C2)}
        self.ov_TCO_cand_map = {(o, v): TCO for (o, v, TCO) in zip(self.cand_obj_ids,
                                                                   self.cand_view_ids,
                                                                   self.cand_TCO)}
        self.residuals_ids = self.make_residuals_ids()

    def make_visibility_matrix(self, cand_view_ids, cand_obj_ids):
        matrix = torch.zeros(self.n_objects, self.n_views, dtype=torch.int, device=self.device)
        matrix[cand_obj_ids, cand_view_ids] = 1
        return matrix

    def make_residuals_ids(self):
        cand_ids, obj_ids, view_ids, point_ids, xy_ids = [], [], [], [], []
        for cand_id in range(self.n_candidates):
            for point_id in range(self.n_points):
                for xy_id in range(2):
                    cand_ids.append(cand_id)
                    obj_ids.append(self.cand_obj_ids[cand_id])
                    view_ids.append(self.cand_view_ids[cand_id])
                    point_ids.append(point_id)
                    xy_ids.append(xy_id)
        residuals_ids = dict(
            cand_id=cand_ids,
            obj_id=obj_ids,
            view_id=view_ids,
            point_id=point_ids,
            xy_id=xy_ids,
        )
        return residuals_ids

    def sample_initial_TWO_TWC(self, seed):
        TWO = torch.zeros(self.n_objects, 4, 4, dtype=self.dtype, device=self.device) * float('nan')
        TWC = torch.zeros(self.n_views, 4, 4, dtype=self.dtype, device=self.device) * float('nan')

        object_to_views = defaultdict(set)
        for v in range(self.n_views):
            for o in range(self.n_objects):
                if self.visibility_matrix[o, v]:
                    object_to_views[o].add(v)

        np_random = np.random.RandomState(seed)
        views_ordered = np_random.permutation(np.arange(self.n_views))
        objects_ordered = np_random.permutation(np.arange(self.n_objects))

        w = views_ordered[0]
        TWC[w] = torch.eye(4, 4, device=self.device, dtype=self.dtype)
        views_initialized = {w, }
        views_to_initialize = set(np.arange(self.n_views)) - views_initialized

        n_pass = 20
        n = 0
        # Initialize views
        while len(views_to_initialize) > 0:
            for v1 in views_ordered:
                if v1 in views_to_initialize:
                    for v2 in views_ordered:
                        if v2 not in views_initialized:
                            continue
                        if (v2, v1) in self.v2v1_TC2C1_map:
                            TC2C1 = self.v2v1_TC2C1_map[(v2, v1)]
                            TWC2 = TWC[v2]
                            TWC[v1] = TWC2 @ TC2C1
                            views_to_initialize.remove(v1)
                            views_initialized.add(v1)
                            break
            n += 1
            if n >= n_pass:
                raise SamplerError('Cannot find an initialization')

        # Initialize objects
        for o in objects_ordered:
            for v in views_ordered:
                if v in object_to_views[o]:
                    TWO[o] = TWC[v] @ self.ov_TCO_cand_map[(o, v)]
                    break
        return TWO, TWC

    @staticmethod
    def extract_pose9d(T):
        T_9d = torch.cat((T[..., :3, :2].transpose(-1, -2).flatten(-2, -1), T[..., :3, -1]), dim=-1)
        return T_9d

    def align_TCO_cand(self, TWO_9d, TCW_9d):
        TWO = compute_transform_from_pose9d(TWO_9d)
        TCW = compute_transform_from_pose9d(TCW_9d)
        TCO = TCW[self.cand_view_ids] @ TWO[self.cand_obj_ids]

        dists, sym = symmetric_distance_reprojected(self.cand_TCO, TCO,
                                                    self.K[self.cand_view_ids],
                                                    self.cand_labels, self.mesh_db)
        TCO_cand_aligned = self.cand_TCO @ sym
        return dists, TCO_cand_aligned

    def forward_jacobian(self, TWO_9d, TCW_9d, residuals_threshold):
        _, TCO_cand_aligned = self.align_TCO_cand(TWO_9d, TCW_9d)

        # NOTE: This could be *much* faster by computing gradients manually, reducing number of operations.
        cand_ids, view_ids, obj_ids, point_ids, xy_ids = [
            self.residuals_ids[k] for k in ('cand_id', 'view_id', 'obj_id', 'point_id', 'xy_id')
        ]

        n_residuals = len(cand_ids)  # Number of residuals
        arange_n = torch.arange(n_residuals)

        TCW_9d = TCW_9d.unsqueeze(0).repeat(n_residuals, 1, 1).requires_grad_()
        TWO_9d = TWO_9d.unsqueeze(0).repeat(n_residuals, 1, 1).requires_grad_()

        TWO = compute_transform_from_pose9d(TWO_9d)
        TCW = compute_transform_from_pose9d(TCW_9d)

        TWO_n = TWO[arange_n, obj_ids]
        TCW_n = TCW[arange_n, view_ids]
        TCO_n = TCW_n @ TWO_n
        K_n = self.K[view_ids]

        TCO_cand_n = TCO_cand_aligned[cand_ids]

        points_n = self.obj_points[obj_ids, point_ids].unsqueeze(1)

        TCO_points_n = project_points(points_n, K_n, TCO_n).squeeze(1)[arange_n, xy_ids]
        TCO_cand_points_n = project_points(points_n, K_n, TCO_cand_n).squeeze(1)[arange_n, xy_ids]

        y = TCO_cand_points_n
        yhat = TCO_points_n
        errors = y - yhat
        residuals = (errors ** 2)
        residuals = torch.min(residuals, torch.ones_like(residuals) * residuals_threshold)

        loss = residuals.mean()
        if torch.is_grad_enabled():
            yhat.sum().backward()

        return errors, loss, TWO_9d.grad, TCW_9d.grad

    def compute_lm_step(self, errors, J, lambd):
        errors = errors.view(errors.numel(), 1)
        A = J.t() @ J + lambd * self.idJ
        b = J.t() @ errors
        # Pinverse is faster on CPU.
        h = torch.pinverse(A.cpu()).cuda() @ b
        return h.flatten()

    def optimize_lm(self, TWO_9d, TCW_9d,
                    optimize_cameras=True,
                    n_iterations=50, residuals_threshold=25,
                    lambd0=1e-3, L_down=9, L_up=11, eps=1e-5):
        # See http://people.duke.edu/~hpgavin/ce281/lm.pdf
        n_params_TWO = TWO_9d.numel()
        n_params_TCW = TCW_9d.numel()
        n_params = n_params_TWO + n_params_TCW
        self.idJ = torch.eye(n_params).to(self.device).to(self.dtype)

        prev_iter_is_update = False
        lambd = lambd0
        done = False
        history = defaultdict(list)
        for n in range(n_iterations):

            if not prev_iter_is_update:
                errors, loss, J_TWO, J_TCW = self.forward_jacobian(TWO_9d, TCW_9d, residuals_threshold)

            history['TWO_9d'].append(TWO_9d)
            history['TCW_9d'].append(TCW_9d)
            history['loss'].append(loss)
            history['lambda'].append(lambd)
            history['iteration'].append(n)

            if done:
                break

            # NOTE: This should not be necessary ?
            with torch.no_grad():
                J = torch.cat((J_TWO.flatten(-2, -1), J_TCW.flatten(-2, -1)), dim=-1)
                h = self.compute_lm_step(errors, J, lambd)
                h_TWO_9d = h[:n_params_TWO].view(self.n_objects, 9)
                h_TCW_9d = h[n_params_TWO:].view(self.n_views, 9)
                TWO_9d_updated = TWO_9d + h_TWO_9d
                if optimize_cameras:
                    TCW_9d_updated = TCW_9d + h_TCW_9d
                else:
                    TCW_9d_updated = TCW_9d

            errors, next_loss, J_TWO, J_TCW = self.forward_jacobian(TWO_9d_updated, TCW_9d_updated, residuals_threshold)

            rho = loss - next_loss
            if rho.abs() < eps:
                done = True
            elif rho > eps:
                TWO_9d = TWO_9d_updated
                TCW_9d = TCW_9d_updated
                loss = next_loss
                lambd = max(lambd / L_down, 1e-7)
                prev_iter_is_update = True
            else:
                lambd = min(lambd * L_up, 1e7)
                prev_iter_is_update = False
        return TWO_9d, TCW_9d, history

    def robust_initialization_TWO_TCW(self, n_init=1):
        TWO_9d_init = []
        TCW_9d_init = []
        dists = []
        for n in range(n_init):
            TWO, TWC = self.sample_initial_TWO_TWC(n)
            TCW = invert_T(TWC)
            TWO_9d, TCW_9d = self.extract_pose9d(TWO), self.extract_pose9d(TCW)
            dists_, _ = self.align_TCO_cand(TWO_9d, TCW_9d)
            TWO_9d_init.append(TWO_9d)
            TCW_9d_init.append(TCW_9d)
            dists.append(dists_.mean())
        best_iter = torch.tensor(dists).argmin()
        return TWO_9d_init[best_iter], TCW_9d_init[best_iter]

    def make_scene_infos(self, TWO_9d, TCW_9d):
        TWO = compute_transform_from_pose9d(TWO_9d)
        TCW = compute_transform_from_pose9d(TCW_9d)
        TWC = invert_T(TCW)
        objects = tc.PandasTensorCollection(
            infos=self.obj_infos,
            TWO=TWO,
        )
        cameras = tc.PandasTensorCollection(
            infos=self.cam_infos,
            TWC=TWC,
            K=self.K
        )
        return objects, cameras

    def convert_history(self, history):
        history['objects'] = []
        history['cameras'] = []
        for n in range(len(history['iteration'])):
            TWO_9d = history['TWO_9d'][n]
            TCW_9d = history['TCW_9d'][n]
            objects, cameras = self.make_scene_infos(TWO_9d, TCW_9d)
            history['objects'].append(objects)
            history['cameras'].append(cameras)
        return history

    def solve(self, sample_n_init=1, **lm_kwargs):
        timer_init = Timer()
        timer_opt = Timer()
        timer_misc = Timer()

        timer_init.start()
        TWO_9d_init, TCW_9d_init = self.robust_initialization_TWO_TCW(n_init=sample_n_init)
        timer_init.pause()

        timer_opt.start()
        TWO_9d_opt, TCW_9d_opt, history = self.optimize_lm(
            TWO_9d_init, TCW_9d_init, **lm_kwargs)
        timer_opt.pause()

        timer_misc.start()
        objects, cameras = self.make_scene_infos(TWO_9d_opt, TCW_9d_opt)
        objects_init, cameras_init = self.make_scene_infos(TWO_9d_init, TCW_9d_init)
        history = self.convert_history(history)
        timer_misc.pause()

        outputs = dict(
            objects_init=objects_init,
            cameras_init=cameras_init,
            objects=objects,
            cameras=cameras,
            history=history,
            time_init=timer_init.stop(),
            time_opt=timer_opt.stop(),
            time_misc=timer_misc.stop(),
        )
        return outputs
