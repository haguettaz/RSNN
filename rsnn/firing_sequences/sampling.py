import torch

from .utils import get_cardinality, get_G


def backward_filtering_forward_sampling(L, N, Tr, B=1):
    """_summary_

    Args:
        L (int): the number of channels/neurons.
        N (int): the length.
        Tr (int): the refractory period, i.e., the minimum number of zeros between two consecutive ones.
        B (int, optional): the number of batches. Defaults to 1.

    Returns:
        firing_sequences (torch.BoolTensor): the firing sequences with shape (B, L, N) or (L, N).
    """
    G = get_G(Tr)

    z = torch.zeros((B * L, N), dtype=int)

    p = torch.zeros((B * L, Tr + 1))
    p[:, 0] = get_cardinality(N - Tr, Tr)
    p[:, 1:] = get_cardinality(N - 2 * Tr - 1, Tr) if N > 2 * Tr + 1 else 1

    z[:, 0] = torch.multinomial(p, 1).squeeze()

    msgb = torch.zeros((B * L, N + 1, Tr + 1))
    msgb[torch.arange(B * L), N, z[:, 0]] = 1.0

    # backward filtering
    for n in range(N, 0, -1):
        # msgb[:, n - 1] = (G @ msgb[:, n].view(-1, Tr + 1, 1)).squeeze()
        msgb[:, n - 1] = torch.tensordot(msgb[:, n], G, dims=([1], [1]))

    # forward sampling
    for n in range(1, N):
        p = G[z[:, n - 1]] * msgb[:, n]
        z[:, n] = torch.multinomial(p, 1).squeeze()

    firing_sequences = torch.zeros((B * L, N), dtype=bool)
    firing_sequences[z == Tr] = True

    if B == 1:
        return firing_sequences

    return firing_sequences.view(B, L, N)
