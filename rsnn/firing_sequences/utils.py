# coding=utf-8

import torch
import torch.nn.functional as F


def get_G(Tr):
    """
    Returns the firing-check matrix.

    Args:
        Tr (int): the refractory period.

    Returns:
        G (torch.FloatTensor): the firing-check matrix.
    """
    G = torch.zeros((Tr + 1, Tr + 1))
    G[1:, :-1] = torch.eye(Tr)
    G[0, [0, -1]] = 1.0
    return G


def get_cardinality(N, Tr, cyclic=False):
    """
    Returns the cardinality of the set of (cyclic) firing sequences.

    Args:
        N (int): the length.
        Tr (int): the refractory period.
        cyclic (bool, optional): cyclic firing sequences if True, otherwise firing sequences. Defaults to False.

    Returns:
        (int): the cardinality.
    """
    G = get_G(Tr)
    if cyclic:
        return torch.matrix_power(G, N).trace().item()

    if N <= Tr + 1:
        return float(N + 1)

    return torch.matrix_power(G, N - Tr).sum().item()


def is_predictable(firing_sequences, Tr):
    """
    Returns True if the firing sequences is predictable.

    Args:
        firing_sequences (torch.BoolTensor): the firing sequence with shape (L, N).
        Tr (int): the refractory period.

    Returns:
        (bool): True if predictable, otherwise False.
    """
    if firing_sequences.dim() != 2:
        raise NotImplementedError(f"can only check one L-channels firing sequences with length N at a time")

    L, N = firing_sequences.size()

    firing_sequences = firing_sequences.double()

    # cyclic condition
    padding1 = F.pad(firing_sequences.view(1, L, N), (0, Tr - 1), mode="circular")
    padding2 = F.pad(firing_sequences.view(1, L, N), (0, Tr), mode="circular")

    # one-to-one correspondence from L-channels windows of length T to [0, (T+1)^L - 1], where T = Tr or Tr + 1.
    filter1 = torch.arange(1, Tr + 1, dtype=torch.double).view(1, 1, Tr) * torch.pow(
        Tr + 1, torch.arange(L, dtype=torch.double)
    ).view(1, L, 1)
    filter2 = torch.arange(1, Tr + 2, dtype=torch.double).view(1, 1, Tr + 1) * torch.pow(
        Tr + 2, torch.arange(L, dtype=torch.double)
    ).view(1, L, 1)

    convolution1 = F.conv1d(padding1, filter1, groups=1).view(1, N)
    convolution2 = F.conv1d(padding2, filter2, groups=1).view(1, N)

    if convolution1.min() < 0 or convolution2.min() < 0:
        raise ValueError(f"Overflow due to a too large L*Tr")

    return convolution1.unique().size() == convolution2.unique().size()


def count_predictable(firing_sequences, Tr):
    """
    Returns the number of predictable firing sequences in the batch.

    Args:
        firing_sequences (torch.BoolTensor): the firing sequence with shape (B, L, N).
        Tr (int): the refractory period.

    Returns:
        (int): the number of predictable firing sequences in the batch
    """
    if firing_sequences.dim() != 3:
        raise NotImplementedError(f"the firing sequences tensor must have shape (B, L, N)")

    B, L, _ = firing_sequences.size()

    firing_sequences = firing_sequences.double()

    # cyclic condition
    padding1 = F.pad(firing_sequences, (0, Tr - 1), mode="circular")
    padding2 = F.pad(firing_sequences, (0, Tr), mode="circular")

    # one-to-one correspondence from L-channels windows of length T to [0, (T+1)^L - 1], where T = Tr or Tr + 1.
    filter1 = torch.arange(1, Tr + 1, dtype=torch.double).view(1, 1, Tr) * torch.pow(
        Tr + 1, torch.arange(L, dtype=torch.double)
    ).view(1, L, 1)
    filter2 = torch.arange(1, Tr + 2, dtype=torch.double).view(1, 1, Tr + 1) * torch.pow(
        Tr + 2, torch.arange(L, dtype=torch.double)
    ).view(1, L, 1)

    convolution1 = F.conv1d(padding1, filter1, groups=1)
    convolution2 = F.conv1d(padding2, filter2, groups=1)

    if convolution1.min() < 0 or convolution2.min() < 0:
        raise ValueError(f"Overflow due to a too large L*Tr")

    return sum([convolution1[b].unique().size() == convolution2[b].unique().size() for b in range(B)])
