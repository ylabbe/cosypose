from cosypose.config import PROJECT_DIR
from torch.utils.cpp_extension import load


def load_extension(optimization='-O3'):
    module = load(name='cosypose_cext',
                  sources=[
                      PROJECT_DIR / 'cosypose/multiview/csrc/ransac.cpp',
                  ],
                  extra_cflags=[optimization],
                  verbose=True)
    return module
