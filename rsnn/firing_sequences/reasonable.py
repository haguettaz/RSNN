import numpy as np

from ..utils.utils import cyclic_range
from .classic import f_set, is_in_f


def is_in_r(seq, length, Tr, Th):

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    if not is_in_f(seq, length, Tr):
        return False

    if Th >= length:  # proof : the sequence is periodic !!
        return True

    indices_1 = cyclic_range(np.arange(length), Th, length)
    indices_2 = cyclic_range(np.arange(length), Th + 1, length)

    unique_1 = np.unique(seq[indices_1], axis=0)
    unique_2 = np.unique(seq[indices_2], axis=0)

    return len(unique_1) == len(unique_2)


def r_set(length, Tr, Th):
    return {seq for seq in f_set(length, Tr) if is_in_r(seq, length, Tr, Th)}


def r_card(length, Tr, Th):
    return len(r_set(length, Tr, Th))
