# coding=utf-8

import torch
import torch.nn.functional as F


def get_G(Tr):
    G = torch.zeros((Tr + 1, Tr + 1))
    G[1:, :-1] = torch.eye(Tr)
    G[0, [0, -1]] = 1.0
    return G


def get_cardinality(N, Tr, cyclic=False):
    G = get_G(Tr)
    if cyclic:
        return torch.matrix_power(G, N).trace().item()

    if N <= Tr + 1:
        return float(N + 1)

    return torch.matrix_power(G, N - Tr).sum().item()


def is_predictable(firing_sequences, Tr):
    B, L, _ = firing_sequences.size()
    if B > 1:
        raise NotImplementedError

    firing_sequences = firing_sequences.double()

    # cyclic condition
    padding1 = F.pad(firing_sequences, (0, Tr - 1), mode="circular")
    padding2 = F.pad(firing_sequences, (0, Tr), mode="circular")

    # one-to-one correspondence from L-channels windows of length T to [0, (T+1)^L - 1], where T = Tr or Tr + 1.
    filter1 = torch.arange(1, Tr + 1, dtype=torch.double)[None, None, :] * torch.pow(Tr + 1, torch.arange(L, dtype=torch.double))[None, :, None]
    filter2 = torch.arange(1, Tr + 2, dtype=torch.double)[None, None, :] * torch.pow(Tr + 2, torch.arange(L, dtype=torch.double))[None, :, None]

    convolution1 = F.conv1d(padding1, filter1, groups=1)
    convolution2 = F.conv1d(padding2, filter2, groups=1)

    if convolution1.min() < 0 or convolution2.min() < 0:
        raise ValueError(f"L and Tr are too large...")

    for b in range(B):
        if convolution1[b].unique().size() != convolution2[b].unique().size():
            return False

    return True


def count_predictable(firing_sequences, Tr):
    B, L, _ = firing_sequences.size()

    firing_sequences = firing_sequences.double()

    # cyclic condition
    padding1 = F.pad(firing_sequences, (0, Tr - 1), mode="circular")
    padding2 = F.pad(firing_sequences, (0, Tr), mode="circular")

    # one-to-one correspondence from L-channels windows of length T to [0, (T+1)^L - 1], where T = Tr or Tr + 1.
    filter1 = torch.arange(1, Tr + 1, dtype=torch.double)[None, None, :] * torch.pow(Tr + 1, torch.arange(L, dtype=torch.double))[None, :, None]
    filter2 = torch.arange(1, Tr + 2, dtype=torch.double)[None, None, :] * torch.pow(Tr + 2, torch.arange(L, dtype=torch.double))[None, :, None]

    convolution1 = F.conv1d(padding1, filter1, groups=1)
    convolution2 = F.conv1d(padding2, filter2, groups=1)

    if convolution1.min() < 0 or convolution2.min() < 0:
        raise ValueError(f"L and Tr are too large...")

    count = 0
    for b in range(B):
        if convolution1[b].unique().size() == convolution2[b].unique().size():
            count += 1

    return count
