import math

import torch


def compute_observation_matrices(spike_sequences, segmentation, delays, sources, Nr, impulse_resp, impulse_resp_deriv):
    """
    Computes the observation matrices C_f (firing times), C_a (active times), and C_s (silent times).

    Args:
        spike_sequences (torch.BoolTensor): spike sequences with size (L, N).
        delays (torch.FloatTensor): delays with size (K).
        sources (torch.IntTensor): sources with size (K).
        Nr (int): refractory period.
        impulse_resp (function): impulse response function.
        impulse_resp_deriv (function): derivative of the impulse response function.

    Returns:
        C_f (torch.FloatTensor): observation tensor for firing times with size (N_f, 2, K).
        C_a (torch.FloatTensor): observation tensor for active times with size (N_a, 1, K).
        C_s (torch.FloatTensor): observation tensor for silent times with size (N_s, 1, K).
    """
    L, N = spike_sequences.size()
    K = sources.size(0)
    M = math.ceil(delays.max() / Nr) + 1

    # compute relative firing times of all neurons to the neuron of interest
    rel_firing_times = torch.ones(K, 2 * M)
    for k in range(K):
        firing_times_k = torch.argwhere(spike_sequences[sources[k]]).flatten()
        N_f_k = firing_times_k.size(0)
        Q, R = 2 * M // N_f_k, 2 * M % N_f_k
        for q in range(Q):
            rel_firing_times[k, q * N_f_k : (q + 1) * N_f_k] = firing_times_k - (q + 1) * N + delays[k]
        rel_firing_times[k, Q * N_f_k :] = firing_times_k[:R] - Q * N + delays[k]

    # compute C_f, observation matrices for firing times (class 0)
    indices_f = torch.argwhere(segmentation == 0).flatten()
    N_f = indices_f.size(0)
    C_f = torch.zeros(N_f, 2, K)
    C_f[:, 0] = impulse_resp(indices_f[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1)
    C_f[:, 1] = impulse_resp_deriv(indices_f[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1)

    # compute C_a, observation matrices for active times (class 1)
    indices_a = torch.argwhere(segmentation == 1).flatten()
    N_a = indices_a.size(0)
    C_a = torch.zeros(N_a, 1, K)
    C_a[:, 0, :] = impulse_resp_deriv(indices_a[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1)

    # compute C_s, observation matrices for silent times (class 2)
    indices_s = torch.argwhere(segmentation == 2).flatten()
    N_s = indices_s.size(0)
    C_s = torch.zeros(N_s, 1, K)
    C_s[:, 0, :] = impulse_resp(indices_s[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1)

    return C_f, C_a, C_s


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
