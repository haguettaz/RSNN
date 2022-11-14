import random

import torch

# def compute_observation_tensor(spike_sequences, delays, origins, Nr, impulse_response):
#     """
#     Returns the observation tensor, i.e., the tensor of individual contributions from the K inputs to the L neurons.

#     Args:
#         spike_sequences (torch.BoolTensor): the firing sequences with shape (N, L).
#         delays (torch.FloatTensor): the transmission delays with shape (L, K).
#         origins (torch.LongTensor): the presynaptic neuron indices with shape (L, K).
#         Nr (int): the refractory period.
#         impulse_response (function): the impulse response.

#     Returns:
#         (torch.FloatTensor): the observation tensor with shape (L, N, 1, K).
#     """
#     L, N = spike_sequences.size()
#     K = delays.size(1)
#     M = math.ceil(delays.max() / Nr)

#     # split tensor of firing signals (inputs) into M+1 receptive fields (channels)
#     input = spike_sequences[origins].float().view(L * K, 1, N).expand(L * K, M + 1, N)

#     # filters to find distance to the last firing, in the M+1 (absolute) receptive fields
#     filters = torch.zeros(M + 1, 1, (M + 1) * Nr + 1)
#     for m in range(M + 1):
#         filters[m, 0, -(m + 1) * Nr - 1 : -m * Nr - 1] = torch.arange((m + 1) * Nr, m * Nr, -1)

#     convolution = F.conv1d(F.pad(input, ((M + 1) * Nr, 0), mode="circular"), filters, groups=M + 1).view(L, K, M + 1, N)
#     convolution[convolution == 0] = torch.nan  # 0 means no firing in the m-th receptive field
#     return impulse_response(convolution - delays[:, :, None, None]).nansum(dim=2, keepdim=True)


def get_input(references, mean_noise, var_noise, T):
    input = []
    for l in range(len(references)):
        input.append([])
        for n in range(len(references[l])):
            noisy_firing_time = min(-1e-9, references[l][n] - T + random.gauss(mean_noise, var_noise))
            input[l].append(noisy_firing_time)
    return input


def compute_drift(firing_times, references, period, num_period):
    """
    Computes the mean and std of the firing times' drifts.

    Args:
        firing_times (list): firing times of the neurons.
        references (list): reference firing times of the neurons.
        period (float): period duration.
        num_period (int): period number.

    Raises:
        ValueError: number of spikes must coincide with the reference.

    Returns:
        (tuple): mean and std of the firing times' drifts.
    """
    L = len(firing_times)

    last_firing_times = [
        [s for s in firing_times[l] if (num_period * period <= s < (num_period + 1) * period)] for l in range(L)
    ]
    means, stds = torch.zeros(L), torch.zeros(L)

    for l in range(L):
        D = len(references[l])
        if len(last_firing_times[l]) != D:
            print(f"The network cannot synchronize with the reference...")
            return torch.nan, torch.nan

        drifts = torch.empty(D, D)
        for d in range(D):
            drifts[d] = (torch.tensor(last_firing_times[l]) - torch.tensor(references[l]).roll(d)) % period

        ref_id = drifts.std(dim=1).argmin()

        means[l] = drifts[ref_id].mean()
        stds[l] = drifts[ref_id].std()

    return means.nanmean().item(), stds.nanmean().item()
