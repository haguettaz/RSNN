# coding=utf-8

import torch
import torch.nn.functional as F


def get_spiking_check_matrix(Nr):
    """Returns the spiking-check matrix G.

    Args:
        Nr (int): refractory length.

    Returns:
        (torch.FloatTensor): the spiking-check matrix.
    """
    G = torch.zeros((Nr + 1, Nr + 1))
    G[1:, :-1] = torch.eye(Nr)
    G[0, [0, -1]] = 1.0
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
