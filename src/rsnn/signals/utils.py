# coding=utf-8

import numpy as np


def norm(x):
    return x / np.sum(x)

def sphere_intersection(array, radius, length, time_step):
    if array.size == 0:
        return np.array([])
    
    shift = array[0] - radius
    shifted_array = array - shift
    array = [np.arange(shifted_array[i] - radius, shifted_array[i] + radius + time_step, time_step) for i in range(shifted_array.size)]
    return (np.concatenate(array) + shift) % length

def sphere_intersection_complement(array, radius, length, time_step):
    if array.size == 0:
        return np.arange(0, length, time_step)

    shift = array[0] - radius
    shifted_array = array - shift
    array = [np.arange(shifted_array[i] + radius + time_step, shifted_array[i+1] - radius, time_step) for i in range(shifted_array.size - 1)]
    array += [np.arange(shifted_array[-1] + radius, length, time_step)]
    return (np.concatenate(array) + shift) % length


# def get_spiking_matrix(Nr, p=None):
#     """
#     Returns the spiking matrix G.

#     Args:
#         Nr (int): refractory length.
#         p (float, optional): spiking probability. Defaults to None.

#     Returns:
#         (np.FloatTensor): the spiking matrix.
#     """
#     G = np.zeros((Nr + 1, Nr + 1))
#     if p is None:
#         G[1:, :-1] = np.eye(Nr) 
#         G[0, 0] = 1
#         G[0, -1] = 1
#     else:
#         G[1:, :-1] = np.eye(Nr)*(1-p)
#         G[0, 0] = 1 - p
#         G[0, -1] = p
#     return G


# def get_phi0(Nr, p=None, tol=1e-12):
#     """
#     Returns the largest eigenvalue of the spike matrix G.
    
#     Args:
#         Nr (int): refractory length.
#         tol (float, optional): stopping criterion tolerance. Defaults to 1e-12.

#     Returns:
#         (float): largest eigenvalue phi0 of the spike matrix G.
#     """
#     if p is None:
#         f = lambda phi_: (phi_**(Nr+1) - phi_**Nr - 1) / ((Nr + 1)*phi_**Nr - Nr * phi_ ** (Nr-1))
#         phi0 = (Nr + 2)**(1 / (Nr + 1))
#     else:    
#         f = lambda phi_: (phi_**(Nr+1) - (1-p)*phi_**(Nr) - p*(1-p)**Nr) / ((Nr + 1)*phi_**Nr - Nr * (1-p)* phi_ ** (Nr-1))
#         phi0 = 1

#     dphi0 = f(phi0)

#     # Newton iteration
#     while abs(dphi0) > tol:
#         phi0 = phi0 - dphi0
#         dphi0 = f(phi0)

#     return phi0
