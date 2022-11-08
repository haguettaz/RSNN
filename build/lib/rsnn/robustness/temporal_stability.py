import math

import torch

from ..utils.utils import mod


def compute_v0(Phi, max_iter=50):
    """_summary_

    Args:
        Phi (_type_): _description_
        max_iter (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    B = Phi.clone()
    for _ in range(max_iter):
        B = B @ B
        B /= B.sum(dim=1, keepdim=True)  # for numerical stability even when max_iter is large

    return B[0]


def compute_jitter_transition_matrices(spike_sequences, weights, delays, sources, Tr, impulse_resp_deriv):
    """_summary_

    Args:
        spike_sequences (_type_): _description_
        weights (_type_): _description_
        delays (_type_): _description_
        sources (_type_): _description_
        Nr (_type_): _description_
        impulse_resp_deriv (_type_): _description_

    Returns:
        _type_: _description_
    """
    N, L = spike_sequences.size()
    K = delays.size(1)

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

    # compute distance to the (M+1)-last firing times in the neurons' referentials (with delays)
    rel_firing_times = abs_firing_times[:, sources] - delays[None, :, :, None]

    jitter_transition_matrices = torch.zeros(N, L * M, L * M)
    for n in range(N):
        silent_neurons = torch.argwhere(~spike_sequences[n]).flatten()
        indices = (M * silent_neurons[None, :] + torch.arange(M)[:, None]).flatten()
        jitter_transition_matrices[n, indices, indices] = 1.0

        if silent_neurons.nelement() == L:
            continue

        active_neurons = torch.argwhere(spike_sequences[n]).flatten()

        # new firing jitter as a function of the previous firing jitter
        in_neurons = sources[active_neurons]
        rows = (active_neurons * M)[:, None]
        cols = (in_neurons[:, :, None] * M + torch.arange(M)[None, None, :]).view(-1, K * M)
        inputs = impulse_resp_deriv(rel_firing_times[n, active_neurons]) * weights[active_neurons][:, :, None]
        jitter_transition_matrices[n, rows, cols] = inputs.view(-1, K * M)

        # old firing jitters are shifted by one time step
        rows = (active_neurons[:, None] * M + torch.arange(1, M)[None, :]).flatten()
        cols = rows - 1
        jitter_transition_matrices[n, rows, cols] = 1.0

    return jitter_transition_matrices / jitter_transition_matrices.sum(dim=-1, keepdim=True)
