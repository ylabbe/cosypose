import numpy as np
import pinocchio as pin
import eigenpy
eigenpy.switchToNumpyArray()


def parse_pose_args(pose_args):
    if len(pose_args) == 2:
        pos, orn = pose_args
        pose = Transform(orn, pos)
    elif isinstance(pose_args, Transform):
        pose = pose_args
    else:
        raise ValueError
    return pose


class Transform:
    def __init__(self, *args):
        if len(args) == 0:
            raise NotImplementedError
        elif len(args) == 7:
            raise NotImplementedError
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Transform):
                self.T = arg.T
            elif isinstance(arg, pin.SE3):
                self.T = arg
            else:
                arg_array = np.asarray(arg)
                if arg_array.shape == (4, 4):
                    R = arg_array[:3, :3]
                    t = arg_array[:3, -1]
                    self.T = pin.SE3(R, t.reshape(3, 1))
                else:
                    raise NotImplementedError
        elif len(args) == 2:
            args_0_array = np.asarray(args[0])
            n_elem_rot = len(args_0_array.flatten())
            if n_elem_rot == 4:
                xyzw = np.asarray(args[0]).flatten().tolist()
                wxyz = [xyzw[-1], *xyzw[:-1]]
                assert len(wxyz) == 4
                q = pin.Quaternion(*wxyz)
                q.normalize()
                R = q.matrix()
            elif n_elem_rot == 9:
                assert args_0_array.shape == (3, 3)
                R = args_0_array
            t = np.asarray(args[1])
            assert len(t) == 3
            self.T = pin.SE3(R, t.reshape(3, 1))
        elif len(args) == 1:
            if isinstance(args[0], Transform):
                raise NotImplementedError
            elif isinstance(args[0], list):
                array = np.array(args[0])
                assert array.shape == (4, 4)
                self.T = pin.SE3(array)
            elif isinstance(args[0], np.ndarray):
                assert args[0].shape == (4, 4)
                self.T = pin.SE3(args[0])

    def __mul__(self, other):
        T = self.T * other.T
        return Transform(T)

    def inverse(self):
        return Transform(self.T.inverse())

    def __str__(self):
        return str(self.T)

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return 7

    def toHomogeneousMatrix(self):
        return self.T.homogeneous

    @property
    def translation(self):
        return self.T.translation.reshape(3)

    @property
    def quaternion(self):
        return pin.Quaternion(self.T.rotation)
