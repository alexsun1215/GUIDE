from typing import List, Iterator

import numpy as np
from torch.utils.data import Sampler

from .dataset import GBCDataset


class HierarchicalBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: GBCDataset, ratio=(4, 2, 2), seed: int = 2026):
        self.ratio = ratio
        self.batch_size = sum(ratio)
        self.rng = np.random.RandomState(seed)
        l1, l2, l3 = [], [], []
        for i, s in enumerate(dataset.samples):
            if s.has_lesion:
                l3.append(i)
            elif s.has_organ:
                l2.append(i)
            else:
                l1.append(i)
        self.pools = [l3, l2, l1]
        self.n = len(dataset)
        self.num_batches = max(1, self.n // self.batch_size)

    def _draw(self, pool_idx: int, k: int) -> List[int]:
        pool = self.pools[pool_idx]
        if len(pool) == 0:
            pool = [x for p in self.pools for x in p]
        replace = len(pool) < k
        return list(self.rng.choice(pool, size=k, replace=replace))

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch: List[int] = []
            for pool_idx, k in enumerate(self.ratio):
                batch.extend(self._draw(pool_idx, k))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
