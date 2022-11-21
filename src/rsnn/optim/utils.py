import torch

def get_obs_mx(neuron_idx, spike_sequences, delays, sources, Tr, eps, impulse_resp, impulse_resp_deriv):
    """
    Computes the observation matrices C_f (firing times), C_a (active times), and C_s (silent times).

    Args:
        spike_sequences (torch.BoolTensor): spike sequences with size (L, N).
        delays (torch.FloatTensor): delays with size (K).
        sources (torch.IntTensor): sources with size (K).
        Tr (int): refractory period.
        eps (int): length of the firing surrounding window.
        impulse_resp (function): impulse response function.
        impulse_resp_deriv (function): derivative of the impulse response function.
        device (torch.device, optional): device on which to put the observation tensors. Defaults to None.

    Returns:
        C_f (torch.FloatTensor): observation tensor for firing times with size (N_f, K).
        C_a (torch.FloatTensor): observation tensor for active times with size (N_a, K).
        C_s (torch.FloatTensor): observation tensor for silent times with size (N_s, K).
    """

    N = spike_sequences.size(1)
    K = sources.size(0)

    indices_f, indices_a, indices_s = parse_ss(spike_sequences[neuron_idx], Tr, eps)
    
    # Observation matrices at firing times
    N_f = indices_f.size(0)
    C_f = torch.empty(N_f, K)
    for k in range(K):
        C_f[:, k] = impulse_resp(
            (indices_f.view(-1, 1) - torch.argwhere(spike_sequences[sources[k]]).view(1, -1) - delays[k]) % N
        ).sum(dim=-1)

    # Observation matrices at active times
    N_a = indices_a.size(0)
    C_a = torch.empty(N_a, K)
    for k in range(K):
        C_a[:, k] = impulse_resp_deriv(
            (indices_a.view(-1, 1) - torch.argwhere(spike_sequences[sources[k]]).view(1, -1) - delays[k]) % N
        ).sum(dim=-1)

    # Observation matrices at silent times
    N_s = indices_s.size(0)
    C_s = torch.empty(N_s, K)
    for k in range(K):
        C_s[:, k] = impulse_resp(
            (indices_s.view(-1, 1) - torch.argwhere(spike_sequences[sources[k]]).view(1, -1) - delays[k]) % N
        ).sum(dim=-1)

    # # compute distance to the M-last firing times in the network referential
    # abs_firing_times = torch.empty(N, L, M)
    # for l in range(L):
    #     tmp = mod((torch.arange(N)[:, None] - torch.argwhere(spike_sequences[:, l]).flatten()[None, :]), N, -1)
    #     if tmp.numel():
    #         tmp_ = (tmp[:, :, None] + N * torch.arange(math.ceil((M) / tmp.size(1)))[None, None, :]).reshape(N, -1)
    #         abs_firing_times[:, l], _ = torch.topk(tmp_, M, dim=1, largest=False)
    #     else:
    #         abs_firing_times[:, l] = -1e9

    # # compute distance to the M-last firing times in the neurons' referentials (with delays)
    # rel_firing_times = abs_firing_times[:, sources] - delays[None, :, :, None]

    # observation_matrices = torch.zeros(N, L, 2, K)
    # observation_matrices[:, :, 0, :] = impulse_resp(rel_firing_times).sum(dim=-1)
    # observation_matrices[:, :, 1, :] = impulse_resp_deriv(rel_firing_times).sum(dim=-1)

    # return observation_matrices

    # # compute relative firing times of all neurons to the neuron of interest
    # rel_firing_times = torch.ones(K, 2 * M)
    # for k in range(K):
    #     firing_times_k = torch.argwhere(spike_sequences[sources[k]]).flatten()
    #     N_f_k = firing_times_k.size(0)
    #     Q, R = 2 * M // N_f_k, 2 * M % N_f_k
    #     for q in range(Q):
    #         rel_firing_times[k, q * N_f_k : (q + 1) * N_f_k] = firing_times_k - (q + 1) * N + delays[k]
    #     rel_firing_times[k, Q * N_f_k :] = firing_times_k[:R] - Q * N + delays[k]

    # # compute C_f, observation matrices for firing times (class 0)
    # indices_f = torch.argwhere(segmentation == 0).flatten()
    # # N_f = indices_f.size(0)
    # C_f = impulse_resp(indices_f[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1).unsqueeze(dim=1)
    # print(C_f.abs().sum(dim=[1, 2]))
    # # C_f = torch.zeros(N_f, 2, K)
    # # C_f[:, 0] = impulse_resp(indices_f[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1)
    # # C_f[:, 1] = impulse_resp_deriv(indices_f[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1)

    # # compute C_a, observation matrices for active times (classes 0 and 1)
    # indices_a = torch.argwhere(segmentation < 2).flatten()
    # # N_a = indices_a.size(0)
    # # C_a = torch.zeros(N_a, 1, K)
    # C_a = impulse_resp_deriv(indices_a[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1).unsqueeze(dim=1)

    # # compute C_s, observation matrices for silent times (class 2)
    # indices_s = torch.argwhere(segmentation == 2).flatten()
    # # N_s = indices_s.size(0)
    # # C_s = torch.zeros(N_s, 1, K)
    # C_s = impulse_resp(indices_s[:, None, None] - rel_firing_times[None, :, :]).sum(dim=-1).unsqueeze(dim=1)

    return C_f, C_a, C_s


def parse_ss(ss, Nr, eps):
    """
    Segment spike sequence into four categories: (0) firing times, (1) active times, (2) silent times, and (3) refractory times.

    Args:
        spike_sequence (torch.BoolTensor): spike sequences with size (N).
        Nr (int): length of the refractory period.
        eps (int): length of the firing surrounding window.

    Returns:
        (torch.ByteTensor): segmentation tensor with size (N)
    """

    N = ss.size(0)

    indices_f = torch.argwhere(ss).flatten()

    # set silent times (class 2)
    sgm = 2 * torch.ones(N, dtype=torch.uint8)

    # set refractory times (class 3)
    indices_r = (indices_f[None, :] + torch.arange(Nr + 1)[:, None]).flatten() % N
    sgm[indices_r] = 3

    # set active times (class 1)
    indices_a = (indices_f[None, :] + torch.arange(-eps, eps + 1)[:, None]).flatten() % N
    sgm[indices_a] = 1

    # set firing times (class 0)
    sgm[indices_f] = 0

    return torch.argwhere(sgm == 0).flatten(), torch.argwhere(sgm < 2).flatten(), torch.argwhere(sgm == 2).flatten()


def test_parse_ss():
    assert parse_ss(
        torch.tensor([False, False, False, False, True, False, False, False, True, False]), 
        2, 
        1
    ) == (torch.tensor([4, 8]), torch.tensor([3, 4, 5, 7, 8, 9]), torch.tensor([1, 2]))


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
