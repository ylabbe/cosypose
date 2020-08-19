from itertools import chain


class MultiEpochDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = None
        self.epoch_id = -1
        self.batch_id = 0
        self.n_repeats_sampler = 1
        self.sampler_length = None
        self.id_in_sampler = None

    def __iter__(self):
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)

            self.sampler_length = len(self.dataloader)
            self.id_in_sampler = 0
            while self.sampler_length <= 2 * self.dataloader.num_workers:
                self.sampler_length += len(self.dataloader)
                next_index_sampler = iter(self.dataloader_iter._index_sampler)
                self.dataloader_iter._sampler_iter = chain(
                    self.dataloader_iter._sampler_iter, next_index_sampler)

        self.epoch_id += 1
        self.batch_id = 0
        self.epoch_size = len(self.dataloader_iter)

        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        if self.batch_id == self.epoch_size:
            raise StopIteration

        elif self.id_in_sampler == self.sampler_length - 2 * self.dataloader.num_workers:
            next_index_sampler = iter(self.dataloader_iter._index_sampler)
            self.dataloader_iter._sampler_iter = next_index_sampler
            self.id_in_sampler = 0

        idx, batch = self.dataloader_iter._get_data()
        self.dataloader_iter._tasks_outstanding -= 1
        self.dataloader_iter._process_data(batch)

        # batch = next(self.dataloader_iter)
        self.batch_id += 1
        self.id_in_sampler += 1
        return batch

    def get_infos(self):
        return dict()

    def __del__(self):
        del self.dataloader_iter
