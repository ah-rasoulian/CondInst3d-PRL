import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader, WeightedRandomSampler
from typing import Iterator, Optional


class DistributedWeightedSampler(Sampler[int]):
    """
    DDP-aware weighted sampler.

    Samples globally according to `weights`, then shards the sampled index list
    across ranks so every process gets a different slice.

    This preserves weighted sampling behavior much better than using a plain
    WeightedRandomSampler independently on each rank.
    """
    def __init__(
        self,
        weights,
        num_samples: int,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

        # num_samples here means samples PER RANK per epoch
        self.num_samples = int(num_samples)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Draw a global list, then shard by rank
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g,
        ).tolist()

        # Rank-specific slice
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch