import torch

from .utils import get_phi0, get_spiking_matrix


def rand_ss(L, N, Nr, p=None):
    """
    Sample a multi-channel spike sequence uniformly at random.

    Args:
        L (int): the number of neurons.
        N (int): the length of the sequence.
        Nr (int): the refractory period of the sequence.
        p (float, optional): the spiking probability. Defaults to 0.5.

    Returns:
        sample (torch.BoolTensor): the spike sequence with shape (L, N).
    """
    if p == 0:
        return torch.zeros((L, N), dtype=torch.bool)        

    i_z = torch.empty((L, N), dtype=torch.long)

    G = get_spiking_matrix(Nr, p) / get_phi0(Nr, p) # Rescale G to have largest eigenvalue 1
    
    # First sample z0 by marginalizing over (z_1, ..., z_{N-1})
    pz = G.matrix_power(N).diag()

    if pz.max() == 0:
        raise ValueError(f"The spiking probability p={p} is not compatible with the refractory period Nr={Nr} and the sequence length N={N}.")

    i_z[:, 0] = torch.multinomial(pz, L, replacement=True)

    # Then sample z_1, ..., z_{N-1} by backward filtering forward sampling
    # 1. Backward filtering
    msgb = torch.zeros((L, N + 1, Nr + 1, 1))
    msgb[torch.arange(L), N, i_z[:, 0]] = 1.0
    for n in range(N, 0, -1):
        msgb[:, n - 1] = G @ msgb[:, n]

    # 2. Forward sampling
    for n in range(1, N):
        pz = G[i_z[:, n - 1]] * msgb[:, n, :, 0]
        i_z[:, n] = torch.multinomial(pz, 1).squeeze()

    # Finally, convert to spike sequences
    sample = torch.zeros((L, N), dtype=torch.bool)
    sample[i_z == Nr] = True
    return sample
