import torch

from .utils import get_phi0, get_spiking_check_matrix


def sample_spike_sequences(L, N, Nr):
    """
    Sample a multi-channel spike sequence uniformly at random.

    Args:
        L (int): the number of neurons.
        N (int): the length.
        Nr (int): the refractory length.

    Returns:
        sample (torch.BoolTensor): the spike sequence with shape (L, N).
    """
    i_z = torch.empty((L, N), dtype=torch.int64)

    # First sample z0 by marginalizing over (z_1, ..., z_{N-1})
    G_ = get_spiking_check_matrix(Nr) / get_phi0(Nr)  # dividing by phi0 prevent the matrix power to explode
    pz0 = G_.matrix_power(N).diag() / G_.matrix_power(N).diag().sum()
    i_z[:, 0] = torch.multinomial(pz0, L, replacement=True)

    # Then sample z_1, ..., z_{N-1} by backward filtering forward sampling
    # 1. Backward filtering
    msgb = torch.zeros((L, N + 1, Nr + 1, 1))
    msgb[torch.arange(L), N, i_z[:, 0]] = 1.0
    for n in range(N, 0, -1):
        msgb[:, n - 1] = G_ @ msgb[:, n]

    # 2. Forward sampling
    for n in range(1, N):
        pzn = G_[i_z[:, n - 1]] * msgb[:, n, :, 0]
        i_z[:, n] = torch.multinomial(pzn, 1).squeeze()

    # Finally, convert to spike sequence
    sample = torch.zeros((L, N), dtype=torch.bool)
    sample[i_z == Nr] = True
    return sample
