import numpy as np

from ..utils.utils import cyclic_range
from .aperiodic import fa_card, fa_set
from .binary import b_card, b_sample, b_set, is_in_b


def is_in_f(seq, length, Tr):
    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if not is_in_b(seq, length):
        return False

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    indices = cyclic_range(np.arange(length), Tr + 1, length)
    return np.all(np.sum(seq[..., indices], axis=-1) <= 1)


def f_card(length, Tr):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if Tr == 0:
        return b_card(length)

    def recursion(length, Tr):
        if length == 1:
            return 1

        if 2 <= length <= Tr:
            return recursion(length - 1, Tr)

        if length == Tr + 1:
            return recursion(length - 1, Tr) + 1 + Tr

        if Tr + 2 <= length <= 2 * Tr + 1:
            return recursion(length - 1, Tr) + 1

        if 2 * Tr + 2 <= length <= 3 * Tr + 2:
            return recursion(length - 1, Tr) + fa_card(length - 2 * Tr - 1, Tr) + Tr

        return (
            recursion(length - 1, Tr)
            + fa_card(length - 2 * Tr - 1, Tr)
            + Tr * fa_card(length - 3 * Tr - 2, Tr)
        )

    return recursion(length, Tr)


def f_set(length, Tr):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if Tr == 0:
        return b_set(length)

    def recursion(length, Tr):
        if length == 1:
            return {(0,)}

        if 2 <= length <= Tr:
            return {seq + (0,) for seq in recursion(length - 1, Tr)}

        if length == Tr + 1:
            seq1 = {seq + (0,) for seq in recursion(length - 1, Tr)}
            seq2 = {(0,) * k + (1,) + (0,) * (Tr - 1 - k) + (0,) for k in range(Tr)}
            seq3 = {(0,) * Tr + (1,)}
            return seq1.union(seq2.union(seq3))

        if Tr + 2 <= length <= 2 * Tr + 1:
            seq1 = {seq + (0,) for seq in recursion(length - 1, Tr)}
            seq2 = {(0,) * (length - 1) + (1,)}
            return seq1.union(seq2)

        if 2 * Tr + 2 <= length <= 3 * Tr + 2:
            seq1 = {seq + (0,) for seq in recursion(length - 1, Tr)}
            seq2 = {
                (0,) * k + (1,) + (0,) * (length - 2 - Tr) + (1,) + (0,) * (Tr - k)
                for k in range(Tr)
            }
            seq3 = {(0,) * Tr + seq + (0,) * Tr + (1,) for seq in fa_set(length - 2 * Tr - 1, Tr)}
            return seq1.union(seq2.union(seq3))

        seq1 = {seq + (0,) for seq in recursion(length - 1, Tr)}

        seq2 = {
            (0,) * k + (1,) + (0,) * Tr + seq + (0,) * Tr + (1,) + (0,) * (Tr - k)
            for k in range(Tr)
            for seq in fa_set(length - 3 * Tr - 2, Tr)
        }
        seq3 = {(0,) * Tr + seq + (0,) * Tr + (1,) for seq in fa_set(length - 2 * Tr - 1, Tr)}
        return seq1.union(seq2.union(seq3))

    return recursion(length, Tr)


from rsnn.firing_sequences.aperiodic import *
from rsnn.firing_sequences.binary import *
from rsnn.firing_sequences.classic import *
from rsnn.firing_sequences.reducible import *


def f_sample(length, Tr, ends_with=tuple()):

    if length < 1:
        raise ValueError(f"length must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    if Tr == 0:
        return b_sample(length, ends_with)

    # TODO: deal with constraints at sampling
    # const = {"start": list or tuple or None, "end": list or tuple or None}
    # the constraint sequences can be of any size

    if len(ends_with) > length:
        raise ValueError("ends_with is not valid")

    if len(ends_with) > 0 and not is_in_f(
        ends_with + (0,) * (length - len(ends_with)), length, Tr
    ):
        raise ValueError("ends_with is not valid")

    if length < Tr + 1:
        if 1 in ends_with:
            raise ValueError(f"ends_with is not combatible with length and Tr")

        return np.zeros(length, int)

    seq_to_idx = lambda seq_: 0 if sum(seq_) < 1 else seq_.index(1) + 1
    idx_to_seq = (
        lambda idx_, l_: (0,) * l_ if idx_ == 0 else (0,) * (idx_ - 1) + (1,) + (0,) * (l_ - idx_)
    )

    if len(ends_with) < Tr:
        dl = Tr - len(ends_with)
        if sum(ends_with) < 1:
            if length <= 2 * Tr + 1:
                p = np.array([fa_card(length - Tr, Tr)] + [1] * dl)
            else:
                p = np.array([fa_card(length - Tr, Tr)] + [fa_card(length - 2 * Tr - 1, Tr)] * dl)

            idx = np.random.choice(dl + 1, p=p / p.sum())
            ends_with = idx_to_seq(idx, dl) + ends_with

        else:
            ends_with = (0,) * dl + ends_with

    s_start = seq_to_idx(ends_with[-Tr:])
    s_end = seq_to_idx(ends_with[:Tr])

    samples = backward_filtering_forward_sampling(
        length - (len(ends_with) - Tr), Tr, s_start, s_end
    )
    return tuple(idx_to_seq(s, Tr)[-1] for s in samples[1:-Tr]) + ends_with


def backward_filtering_forward_sampling(length, Tr, s_start, s_end):
    def get_A(Tr):
        A = np.zeros((Tr + 1, Tr + 1))
        A[:-1, 1:] = np.identity(Tr)
        A[0, 0] = 1.0
        A[-1, 0] = 1.0
        return A

    def backward_filtering(mu_b, A):
        for i in range(mu_b.shape[0] - 1, 0, -1):
            mu_b[i - 1] = mu_b[i] @ A
        return mu_b

    def forward_sampling(mu_b, A, Tr, s_start):
        samples = [s_start]
        for i in range(mu_b.shape[0] - 1):
            p = 1 / mu_b[i, samples[i]] * mu_b[i + 1] * A[:, samples[i]]
            s = np.random.choice(Tr + 1, p=p)
            samples.append(s)
        return samples

    # message transmission matrix
    A = get_A(Tr)

    # init backward messages
    mu_b = np.zeros((length + 1, Tr + 1))
    mu_b[-1] = np.array([0] * s_end + [1] + [0] * (Tr - s_end))

    # backward filtering from right to left
    mu_b = backward_filtering(mu_b, A)

    # forward sampling from left to right
    return forward_sampling(mu_b, A, Tr, s_start)
