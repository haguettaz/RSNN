import torch


def segment_spike_sequence(spike_sequence, Nr, eps):
    """
    Segment spike sequence into four categories: (0) firing times, (1) active times, (2) silent times, and (3) refractory times.

    Args:
        spike_sequence (torch.BoolTensor): spike sequences with size (N).
        Nr (int): length of the refractory period.
        eps (int): length of the firing surrounding window.

    Returns:
        (torch.ByteTensor): segmentation tensor with size (N)
    """

    N = spike_sequence.size(0)

    firing_indices = torch.argwhere(spike_sequence).flatten()

    # set silent times (class 2)
    segmentation = 2 * torch.ones(N, dtype=torch.uint8)

    # set refractory times (class 3)
    indices = (firing_indices[None, :] + torch.arange(Nr + 1)[:, None]).flatten() % N
    segmentation[indices] = 3

    # set active times (class 1)
    indices = (firing_indices[None, :] + torch.arange(-eps, eps + 1)[:, None]).flatten() % N
    segmentation[indices] = 1

    # set firing times (class 0)
    segmentation[firing_indices] = 0

    return segmentation
