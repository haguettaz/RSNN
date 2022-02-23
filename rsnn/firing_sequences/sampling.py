import numpy as np

from .counting import card
from .utils import is_firing


def multi_channel_sampling(N, L, Tr, ends_with=None):
    samples = np.empty((N, L), int)

    if ends_with is not None:
        if ends_with.shape[0] != N:
            raise ValueError(f"ends_with is not valid (check channels)")
        for n in range(N):
            samples[n] = single_channel_sampling(L, Tr, ends_with[n])
        return samples

    for n in range(N):
        samples[n] = single_channel_sampling(L, Tr)
    return samples


def single_channel_sampling(L, Tr, ends_with=None):
    def seq_to_idx(seq):
        if sum(seq) < 1:
            return 0
        return int(np.argwhere(seq)) + 1

    def idx_to_seq(idx, l):
        if idx == 0:
            return np.zeros(l, int)
        return np.array([0] * (idx - 1) + [1] + [0] * (l - idx))

    if L < 1:
        raise ValueError(f"L must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    def sample_end(L, Tr, ends_with):
        if ends_with is None:
            ends_with = np.array([], int)

        L_end = ends_with.shape[0]
        if L_end > L:
            raise ValueError("ends_with is not valid (check length)")

        if not is_firing(np.concatenate((np.zeros(L - L_end, int), ends_with)), 1, L, Tr):
            raise ValueError("ends_with is not valid (check refratory period)")

        if L_end < Tr:
            dl = Tr - L_end
            if sum(ends_with) < 1:
                if L <= 2 * Tr + 1:
                    p = np.array([card(1, L - Tr, Tr, cyclic=False)] + [1] * dl)
                else:
                    p = np.array(
                        [card(1, L - Tr, Tr, cyclic=False)]
                        + [card(1, L - 2 * Tr - 1, Tr, cyclic=False)] * dl
                    )

                idx = np.random.choice(dl + 1, p=p / p.sum())
                ends_with = np.concatenate((idx_to_seq(idx, dl), ends_with))

            else:
                ends_with = np.concatenate((np.zeros(dl, int), ends_with))

        return ends_with

    ends_with = sample_end(L, Tr, ends_with)

    s_start = seq_to_idx(ends_with[-Tr:])
    s_end = seq_to_idx(ends_with[:Tr])

    samples = backward_filtering_forward_sampling(
        L - (ends_with.shape[0] - Tr), Tr, s_start, s_end
    )
    return np.concatenate((np.array([idx_to_seq(s, Tr)[-1] for s in samples[1:-Tr]]), ends_with))


def backward_filtering_forward_sampling(L, Tr, s_start, s_end):
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

    # message passing matrix
    A = get_A(Tr)

    # init backward messages
    mu_b = np.zeros((L + 1, Tr + 1))
    mu_b[-1] = np.array([0] * s_end + [1] + [0] * (Tr - s_end))

    # backward filtering from right to left
    mu_b = backward_filtering(mu_b, A)

    # forward sampling from left to right
    return forward_sampling(mu_b, A, Tr, s_start)
