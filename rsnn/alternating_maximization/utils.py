import torch
import torch.nn.functional as F


def compute_observation_matrix(firing_sequences, taus, origins, Tr, impulse_response):
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


def check_parameters(wlim, theta, eta, eps, Tr):
    if eps >= Tr:
        raise ValueError(f"eps should be strictly smaller than Tr, but got {eps} >= {Tr}")

    if wlim[0] >= wlim[1]:
        raise ValueError(f"wmin should be strictly smaller than wmax, but got {wlim[0]} >= {wlim[1]}")


def get_mask_at_firing(firing_sequences):
    return firing_sequences


def get_mask_around_firing(firing_sequences, eps):
    L, N = firing_sequences.size()
    filter = torch.ones(L, 1, 2 * eps + 1)
    padding = F.pad(firing_sequences[None, ...].float(), (eps, eps), mode="circular")
    return F.conv1d(padding, filter, groups=L).bool().view(L, N)


def get_mask_refractory_period(firing_sequences, Tr, eps):
    L, N = firing_sequences.size()
    filter = torch.FloatTensor([1] * (Tr - eps) + [0] * (eps + 1)).expand(L, 1, Tr + 1)
    padding = F.pad(firing_sequences[None, ...].float(), (Tr, 0), mode="circular")
    return F.conv1d(padding, filter, groups=L).bool().view(L, N)