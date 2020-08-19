import numpy as np

from .transform import Transform
from .rotations import euler2quat


def make_bop_symmetries(dict_symmetries, n_symmetries_continuous=8, scale=0.001):
    # Note: See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
    sym_discrete = dict_symmetries.get('symmetries_discrete', [])
    sym_continous = dict_symmetries.get('symmetries_continuous', [])
    all_M_discrete = [Transform((0, 0, 0, 1), (0, 0, 0))]
    all_M_continuous = []
    all_M = []
    for sym_n in sym_discrete:
        M = np.array(sym_n).reshape(4, 4)
        M[:3, -1] *= scale
        M = Transform(M)
        all_M_discrete.append(M)
    for sym_n in sym_continous:
        assert np.allclose(sym_n['offset'], 0)
        axis = np.array(sym_n['axis'])
        assert axis.sum() == 1
        for n in range(n_symmetries_continuous):
            euler = axis * 2 * np.pi * n / n_symmetries_continuous
            q = euler2quat(euler)
            M = Transform(q, (0, 0, 0))
            all_M_continuous.append(M)
    for sym_d in all_M_discrete:
        if len(all_M_continuous) > 0:
            for sym_c in all_M_continuous:
                M = sym_c * sym_d
                all_M.append(M.toHomogeneousMatrix())
        else:
            all_M.append(sym_d.toHomogeneousMatrix())
    return np.array(all_M)
