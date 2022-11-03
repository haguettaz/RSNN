import math

import torch

from ..utils.utils import mod


def update(Vw, mw, Vy, my):
    N = Vy.size(0)
    for n in range(N):
        Vw = Vy[n] @ Vw @ Vy.mT[n]
        mw = Vy[n] @ Vw @ my[n]


def compute_observation_matrices(spike_sequences, delays, sources, Tr, impulse_resp, impulse_resp_deriv):
    """_summary_

    Args:
        spike_sequences (_type_): _description_
        delays (_type_): _description_
        origins (_type_): _description_
        M (_type_): _description_
        impulse_resp (_type_): _description_
        impulse_resp_deriv (_type_): _description_

    Returns:
        _type_: _description_
    """
    N, L = spike_sequences.size()
    K = sources.size(1)

    M = math.ceil(delays.max() / Tr) + 1

    # compute distance to the M-last firing times in the network referential
    abs_firing_times = torch.empty(N, L, M)
    for l in range(L):
        tmp = mod((torch.arange(N)[:, None] - torch.argwhere(spike_sequences[:, l]).flatten()[None, :]), N, -1)
        if tmp.numel():
            tmp_ = (tmp[:, :, None] + N * torch.arange(math.ceil((M) / tmp.size(1)))[None, None, :]).reshape(N, -1)
            abs_firing_times[:, l], _ = torch.topk(tmp_, M, dim=1, largest=False)
        else:
            abs_firing_times[:, l] = -1e9

    # compute distance to the M-last firing times in the neurons' referentials (with delays)
    rel_firing_times = abs_firing_times[:, sources] - delays[None, :, :, None]

    observation_matrices = torch.zeros(N, L, 2, K)
    observation_matrices[:, :, 0, :] = impulse_resp(rel_firing_times).sum(dim=-1)
    observation_matrices[:, :, 1, :] = impulse_resp_deriv(rel_firing_times).sum(dim=-1)

    return observation_matrices


# def get_indices_around_firing(spike_sequences, eps):
#     """
#     Returns a mask with True value inside any eps-sphere around a firing time.

#     Args:
#         spike_sequences (torch.BoolTensor): the firing sequences with shape (N, L).
#         eps (int): the radius of the sphere.

#     Returns:
#         times (torch.LongTensor): the times in the eps-sphere of some firing times.
#         neurons (torch.LongTensor): the neurons in the eps-sphere of some firing times.
#     """
#     N = spike_sequences.size(0)
#     indices = torch.argwhere(spike_sequences)
#     times = (indices[:, 0][:, None] + torch.arange(-eps, eps + 1)[None, :]) % N
#     neurons = indices[:, 1][:, None]
#     return times, neurons


# def get_indices_refractory_period(spike_sequences, Nr):
#     """
#     Returns a mask with True value during the refractory period.

#     Args:
#         spike_sequences (torch.BoolTensor): the firing sequences with shape (N, L).
#         Nr (int): the refractory period.

#     Returns:
#         times (torch.LongTensor): the times in the eps-sphere of some firing times.
#         neurons (torch.LongTensor): the neurons in the eps-sphere of some firing times.
#     """
#     N = spike_sequences.size(0)
#     indices = torch.argwhere(spike_sequences)
#     times = (indices[:, 0][:, None] + torch.arange(1, Nr + 1)[None, :]) % N
#     neurons = indices[:, 1][:, None]
#     return times, neurons


# def sequences_to_firing_sequences(sequences, Nr):
#     """
#     Converts binary sequences to firing sequences with a given refractory period.

#     Args:
#         sequences (torch.BoolTensor): the sequences to convert.
#         Nr (int): the refractory period.

#     Returns:
#         spike_sequences (torch.BoolTensor): the firing sequences.
#     """
#     N = sequences.size(-1)
#     spike_sequences = torch.empty_like(sequences)

#     for n in range(N):
#         spike_sequences[:, n] = (spike_sequences[:, max(0, n - Nr) : n].sum(dim=-1) == 0) * sequences[:, n]

#     return spike_sequences
