import numpy as np

from ..utils.utils import cyclic_range
from .classic import f_set, is_in_f


def is_in_a(seq, length, Tr, Th):

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    if not is_in_f(seq, length, Tr):
        return False

    if np.sum(seq) == 0:
        return True

    indices = cyclic_range(np.arange(length), Th, length)
    return np.all(np.sum(seq[..., indices], axis=-1) > 0)


def a_set(length, Tr, Th):
    return {seq for seq in f_set(length, Tr) if is_in_a(seq, length, Tr, Th)}


def a_card(length, Tr, Th):
    return len(a_set(length, Tr, Th))
