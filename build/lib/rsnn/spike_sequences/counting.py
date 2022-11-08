import torch

from .utils import get_phi0, get_spiking_check_matrix


def get_cardinality(N, Nr, approx=True):
    """
    Returns the cardinality of the set of periodic firing sequences.

    Args:
        N (int): the length.
        Nr (int): the refractory period.

    Returns:
        (int): the cardinality.
    """

    if approx:
        phi0 = get_phi0(Nr)
        return phi0 ** N

    G = get_spiking_check_matrix(Nr)
    return torch.matrix_power(G, N).trace().item()
