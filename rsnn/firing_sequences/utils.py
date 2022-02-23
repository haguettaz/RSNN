# coding=utf-8

import numpy as np

from ..utils.utils import cyclic_range


def is_firing(seq, N, L, Tr):
    if N < 1:
        raise ValueError(f"N must be stricly positive")

    if L < 1:
        raise ValueError(f"L must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if N == 1 and seq.shape[0] != L:
        return False

    if N > 1 and seq.shape != (N, L):
        return False

    # check if the sequence is binary
    if seq.size != np.count_nonzero((seq == 0) | (seq == 1)):
        return False

    # check if the refractory period is respected
    indices = cyclic_range(np.arange(L), Tr + 1, L)
    return np.all(np.sum(seq[..., indices], axis=-1) <= 1)


def is_predictable(seq, N, L, Tr, Tw):
    if N < 1:
        raise ValueError(f"N must be stricly positive")

    if L < 1:
        raise ValueError(f"L must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    # check if the dimensions are correct
    if seq.shape != (N, L):
        return False

    # check if the sequence is binary
    if seq.size != np.count_nonzero((seq == 0) | (seq == 1)):
        return False

    if Tw <= Tr:
        raise ValueError(f"Tw must be stricly larger than Tr")

    if Tw >= L:  # proof : the sequence is periodic !!
        return True

    indices_1 = cyclic_range(np.arange(L), Tw, L)
    indices_2 = cyclic_range(np.arange(L), Tw + 1, L)

    unique_1 = np.unique(seq[:, indices_1], axis=1)
    unique_2 = np.unique(seq[:, indices_2], axis=1)

    return unique_1.shape[1] == unique_2.shape[1]

