import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_samples=None, replacement=True, seed=0):
        super().__init__()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.seed = seed

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # total samples per epoch (global). Default = dataset size
        self.num_samples = int(num_samples) if num_samples is not None else len(self.weights)
        # samples per rank
        self.num_samples_per_rank = int(math.ceil(self.num_samples / self.world_size))

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # draw global indices then shard
        total = self.num_samples_per_rank * self.world_size
        indices = torch.multinomial(self.weights, total, self.replacement, generator=g).tolist()

        # shard by rank
        indices = indices[self.rank:total:self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank
