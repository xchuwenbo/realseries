# -*- coding: utf-8 -*-
"""Segment function.
"""
# Author: Wenkai Li

import numpy as np


def slice_generator(series_length, batch_size, discard_last_batch=False):
    """Generate slices for series-like data

    Args:
        series_length (int): Series length.
        batch_size (int): Batch size.
        discard_last_batch (bool, optional): If the last batch not complete, ignore it. Defaults to False.

    Yields:
        slice
    """
    start = 0
    end = (series_length // batch_size) * batch_size
    while start < end:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not discard_last_batch and start < series_length:
        yield slice(start, series_length, 1)


class BatchSegment:
    """[summary]

    Args:
        series_length (int): Series length.
        window_size (int): Window size.
        batch_size (int): Batch size.
        shuffle (bool, optional): Defaults to False.
        discard_last_batch (bool, optional): If the last batch not complete, ignore it. Defaults to False.

    Raises:
        ValueError: Window_size must larger than 1.
        ValueError: Window_size must smaller than series_length

    """
    def __init__(self,
                 series_length,
                 window_size,
                 batch_size,
                 shuffle=False,
                 discard_last_batch=False):
        # check params
        if window_size < 1:
            raise ValueError('window_size must larger than 1.')
        if window_size > series_length:
            raise ValueError('window_size must smaller than series_length')

        indices = np.arange(series_length)
        self._indices = indices.reshape([-1, 1])

        self._offsets = np.arange(-window_size + 1, 1)

        self._series_length = series_length
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._discard_last_batch = discard_last_batch

    def get_iterator(self, arrays):
        """Get data iterator for input sequences.

        Args:
            arrays (list): Contain the data to be iterated, which with the same length.

        Yields:
            tuple: Contain the sliding window data, which has the same order as param: arrays.
        """
        arrays = tuple(np.asarray(a) for a in arrays)

        if self._shuffle:
            np.random.shuffle(self._indices)

        for s in slice_generator(
            series_length=len(self._indices),
            batch_size=self._batch_size,
            discard_last_batch=self._discard_last_batch
        ):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] for a in arrays)
            # print(idx)
            # print(arrays)
            # print(arrays[0][idx])


if __name__ == '__main__':
    a = [i for i in range(0, 300)]
    gen = BatchSegment(300, 30, 256, shuffle=True, discard_last_batch=True).get_iterator([a])
    for item in gen:
        print(item)