import numpy as np


def is_in_b(seq, length):
    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if len(seq) != length:
        return False

    return all(b in {0, 1} for b in seq)


def b_card(length):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    return int(2 ** length)


def b_set(length):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    def recursion(length):
        if length == 1:
            return {(0,), (1,)}

        set0 = {seq + (0,) for seq in recursion(length - 1)}
        set1 = {seq + (1,) for seq in recursion(length - 1)}

        return set0.union(set1)

    return recursion(length)


def b_sample(length):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    return np.random.binomial(1, 0.5, length)
