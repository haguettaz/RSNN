import torch
import torch.nn.functional as F


def compute_observation_matrix(firing_sequences, taus, origins, Tr, impulse_response):
    """_summary_

    Args:
        firing_sequences (torch.BoolTensor): the firing sequences with shape (L, N).
        taus (torch.LongTensor): the transmission delays with shape (L, K).
        origins (torch.LongTensor): the presynaptic neuron indices with shape (L, K).
        Tr (int): the refractory period.
        impulse_response (function): the impulse response (synapse + neuron).

    Returns:
        C (torch.FloatTensor): the observation tensor with shape (L, N, 2, K).
    """
    L, N = firing_sequences.size()
    K = taus.size(1)
    tau_max = taus.max()

    filter = impulse_response(torch.arange(tau_max + Tr, -1, -1)[None, None, :] - taus[:, :, None]).view(L * K, 1, -1)
    input = firing_sequences[origins].float().view(1, L * K, N)
    padded_input = F.pad(input, (Tr + tau_max, 0), mode="circular")
    convolve_input = F.conv1d(padded_input, filter, groups=L * K).reshape(L, K, N).permute(0, 2, 1)

    C = torch.zeros(L, N, 2, K)
    C[..., 0, :] = convolve_input
    C[..., 1, :] = convolve_input - torch.roll(convolve_input, shifts=1, dims=1)

    return C


def get_mask_at_firing(firing_sequences):
    """
    Return a mask with True value at firing times.

    Args:
        firing_sequences (torch.BoolTensor): the firing sequences with shape (L, N).

    Returns:
        (torch.BoolTensor): the at firing mask.
    """
    return firing_sequences


def get_mask_around_firing(firing_sequences, eps):
    """
    Return a mask with True value inside any eps-sphere around a firing time.

    Args:
        firing_sequences (torch.BoolTensor): the firing sequences.
        eps (int): the around parameter.

    Returns:
        (torch.BoolTensor): the around firing mask.
    """
    L, N = firing_sequences.size()
    filter = torch.ones(L, 1, 2 * eps + 1)
    padding = F.pad(firing_sequences[None, ...].float(), (eps, eps), mode="circular")
    return F.conv1d(padding, filter, groups=L).bool().view(L, N)


def get_mask_refractory_period(firing_sequences, Tr, eps):
    """
    Return a mask with True value during the refractory period (and outside of any eps-sphere around a firing time).

    Args:
        firing_sequences (torch.BoolTensor): the firing sequences.
        Tr (int): the refractory period.
        eps (int): the around parameter.

    Returns:
        (torch.BoolTensor): the refractory period mask.
    """
    L, N = firing_sequences.size()
    filter = torch.FloatTensor([1] * (Tr - eps) + [0] * (eps + 1)).expand(L, 1, Tr + 1)
    padding = F.pad(firing_sequences[None, ...].float(), (Tr, 0), mode="circular")
    return F.conv1d(padding, filter, groups=L).bool().view(L, N)
