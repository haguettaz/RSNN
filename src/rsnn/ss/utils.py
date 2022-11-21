# coding=utf-8

import torch
import torch.nn.functional as F


def get_spiking_matrix(Nr, p=None):
    """
    Returns the spiking matrix G.

    Args:
        Nr (int): refractory length.
        p (float, optional): spiking probability. Defaults to None.

    Returns:
        (torch.FloatTensor): the spiking matrix.
    """
    G = torch.zeros((Nr + 1, Nr + 1))
    G[1:, :-1] = torch.eye(Nr)
    G[0, 0] = 1 if p is None else 1 - p
    G[0, -1] = 1 if p is None else p
    return G


def get_phi0(Nr, tol=1e-12):
    """
    Returns the largest eigenvalue of the spike matrix G.
    
    Args:
        Nr (int): refractory length.
        tol (float, optional): stopping criterion tolerance. Defaults to 1e-12.

    Returns:
        (float): largest eigenvalue phi0 of the spike matrix G.
    """
    f = lambda phi_: (phi_ - 1 - phi_ ** (-Nr)) / (Nr + 1 - Nr * phi_ ** (-1))

    # Initialization with the tight upper bound
    phi0 = (Nr + 2) ** (1 / (Nr + 1))
    dphi0 = f(phi0)

    # Newton iteration
    while abs(dphi0) > tol:
        phi0 = phi0 - dphi0
        dphi0 = f(phi0)

    return phi0

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
