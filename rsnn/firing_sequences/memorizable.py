import numpy as np

from .autonomous import a_set, is_in_a
from .reasonable import is_in_r, r_set


# we made the hypothesis that any sequence that is autonomous and reasonable is memorizable!
def is_in_m(seq, length, Tr, Th):

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    return is_in_a(seq, length, Tr, Th) and is_in_r(seq, length, Tr, Th)


def m_set(length, Tr, Th):
    return a_set(length, Tr, Th).intersection(r_set(length, Tr, Th))


def m_card(length, Tr, Th):
    return len(m_set(length, Tr, Th))
