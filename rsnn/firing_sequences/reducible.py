import numpy as np

from ..utils.utils import divisors, is_prime

from .classic import f_set, f_card, is_in_f


def is_in_d(seq, length, Tr):

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    if not is_in_f(seq, length, Tr):
        return False

    for d in divisors(length)[1:]:
        T = length // d
        if np.all([np.array_equal(seq[:T], seq[(k + 1) * T : (k + 2) * T]) for k in range(d - 1)]):
            return True

    return False


def d_card(length, Tr):
    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    def recursion(length, Tr):
        if length == 1:
            return f_card(length, Tr)

        if is_prime(length):
            return f_card(length, Tr) - f_card(1, Tr)

        count = f_card(length, Tr)
        for d in divisors(length)[:-1]:
            count -= recursion(d, Tr)
        return count

    return sum([recursion(d, Tr) for d in divisors(length)[:-1]])


def d_set(length, Tr):
    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    set_ = set()
    for d in divisors(length)[:-1]:
        set_ = set_.union({seq * (length // d) for seq in f_set(d, Tr)})

    return set_


def d_sample(length, Tr):
    raise NotImplementedError
