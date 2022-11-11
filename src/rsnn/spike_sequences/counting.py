import torch

from .utils import get_phi0, get_spiking_matrix


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
        return get_phi0(Nr) ** N

    return get_spiking_matrix(Nr).matrix_power(N).trace().item()
