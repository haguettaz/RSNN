import torch


def parse_spike_sequences(spike_sequences, Nr, eps):
    """
    Segments spike sequences into three categories: (0) refractory period, (1) silent period, and (2) active period.

    Args:
        spike_sequences (torch.BoolTensor): spike sequences with size (L, N).
        Nr (int): length of the refractory period.
        eps (int): length of the firing surrounding window.

    Returns:
        (torch.LongTensor): firing indices.
        (torch.LongTensor): active period indices.
        (torch.LongTensor): silent period indices.
    """
    # init as if all neurons are always in silent period (class 1)
    L, N = spike_sequences.size()
    segmentation = torch.ones(L, N, dtype=torch.long)
    firing_indices = torch.argwhere(spike_sequences)

    # set refractory period (class 0)
    times = (firing_indices[:, 0][None, :] + torch.arange(Nr + 1)[:, None]) % N
    neurons = firing_indices[:, 1]
    segmentation[times, neurons] = 0

    # set firing period (class 2)
    times = (firing_indices[:, 0][None, :] + torch.arange(-eps, eps + 1)[:, None]) % N
    neurons = firing_indices[:, 1]
    segmentation[times, neurons] = 2

    return firing_indices, torch.argwhere(segmentation == 2), torch.argwhere(segmentation == 1)
