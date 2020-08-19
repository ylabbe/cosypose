import sys
import functools
from tqdm import tqdm


def patch_tqdm():
    tqdm = sys.modules['tqdm'].tqdm
    sys.modules['tqdm'].tqdm = functools.partial(tqdm, file=sys.stdout)
    return
