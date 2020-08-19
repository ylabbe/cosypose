from pathlib import Path
import torch
from collections import defaultdict

from cosypose.utils.distributed import get_world_size, get_rank


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.datas = defaultdict(list)

    def add(self, pred_data, gt_data):
        raise NotImplementedError

    def is_data_valid(self, data):
        raise NotImplementedError

    def gather_distributed(self, tmp_dir):
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        rank, world_size = get_rank(), get_world_size()
        tmp_file_template = (tmp_dir / 'rank={rank}.pth.tar').as_posix()
        if rank > 0:
            tmp_file = tmp_file_template.format(rank=rank)
            torch.save(self.datas, tmp_file)

        if world_size > 1:
            torch.distributed.barrier()

        if rank == 0 and world_size > 1:
            all_datas = self.datas
            for n in range(1, world_size):
                tmp_file = tmp_file_template.format(rank=n)
                datas = torch.load(tmp_file)
                for k in all_datas.keys():
                    all_datas[k].extend(datas.get(k, []))
                Path(tmp_file).unlink()
            self.datas = all_datas

        if world_size > 1:
            torch.distributed.barrier()
        return
