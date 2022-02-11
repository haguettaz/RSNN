import numpy as np

from .binary import b_card, b_sample, b_set, is_in_b


def is_in_fa(seq, length, Tr):
    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if not is_in_b(seq, length):
        return False

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    indices = np.arange(length - Tr)[:, None] + np.arange(Tr + 1)[None, :]
    return np.all(np.sum(seq[..., indices], axis=-1) <= 1)


def fa_card(length, Tr):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if Tr == 0:
        return b_card(length)

    def recursion(length, Tr):
        if length == 1:
            return 2

        if 2 <= length <= Tr + 1:
            return recursion(length - 1, Tr) + 1

        return recursion(length - 1, Tr) + recursion(length - Tr - 1, Tr)

    return recursion(length, Tr)


def fa_set(length, Tr):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if Tr == 0:
        return b_set(length)

    def recursion(length, Tr):
        if length == 1:
            return {(0,), (1,)}

        if length <= Tr + 1:
            set1 = {(0,) * (length - 1) + (1,)}
            set2 = {seq + (0,) for seq in recursion(length - 1, Tr)}
            return set1.union(set2)

        set1 = {seq + (0,) for seq in recursion(length - 1, Tr)}
        set2 = {seq + (0,) * Tr + (1,) for seq in recursion(length - Tr - 1, Tr)}
        return set1.union(set2)

    return recursion(length, Tr)


def fa_sample(length, Tr):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if Tr == 0:
        return b_sample(length)

    ...
