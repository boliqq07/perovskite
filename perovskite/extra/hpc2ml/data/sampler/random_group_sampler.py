# -*- coding: utf-8 -*-
# @Time  : 2022/10/18 23:06
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

from typing import Iterator, Optional, Sized

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler


class RandomGroupSampler(Sampler[int]):
    r"""Samples elements in group randomly.
    If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
        group (int): group size.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, group=2) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples  # old
        self.generator = generator
        self.group = group
        super(RandomGroupSampler, self).__init__(data_source=data_source)

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:  # half
        # dataset size might change at runtime
        if self._num_samples is None:
            assert len(self.data_source) % self.group == 0, f"Just For data size can be divided evenly by {self.group}."
            return len(self.data_source) // 2
        else:
            assert self._num_samples % self.group == 0, f"Just For data size can be divided evenly by {self.group}."
            return self._num_samples // 2

    def __iter__(self) -> Iterator[int]:

        def back(arr: torch.Tensor):
            arrs = [torch.sub((arr + 1) * self.group, self.group - i) for i in range(self.group)]
            arrs = torch.vstack(arrs).T
            arr_all = torch.ravel(arrs)
            return arr_all

        n = len(self.data_source) // 2
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from back(torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator)).tolist()
            yield from back(
                torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator)).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from back(torch.randperm(n, generator=generator)).tolist()
            yield from back(torch.randperm(n, generator=generator)).tolist()[:self.num_samples % n]

    def __len__(self) -> int:  # half
        return self.num_samples


def train_test_split_2_group(dataset, train_size=0.80, test_size=0.20, shuffle=True, random_state=None):
    """keep group 2 in old rank."""
    train_index_raw, test_index_raw = train_test_split(np.arange(len(dataset) // 2), train_size=train_size,
                                                       test_size=test_size, shuffle=shuffle, random_state=random_state)

    train_index1 = 2 * train_index_raw
    test_index1 = 2 * test_index_raw

    train_index2 = 2 * train_index_raw + 1
    test_index2 = 2 * test_index_raw + 1

    train_index = np.vstack([train_index1, train_index2]).T.ravel()
    test_index = np.vstack([test_index1, test_index2]).T.ravel()

    train_dataset = dataset[train_index.tolist()]
    test_dataset = dataset[test_index.tolist()]
    return train_dataset, test_dataset
