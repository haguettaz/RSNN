import numpy as np

# from ..rsnn.network import Network
#from ..spike_train.spike_train import PeriodicSpikeTrain


# def get_phi(spike_train, network):
#     """
#     Returns the Phi matrix corresponding to the spike train and the network.

#     Args:
#         spike_train (PeriodicSpikeTrain): The spike train.
#         network (Network): The network.

#     Returns:
#         (np.ndarray): The Phi matrix.
#     """
#     firing_times = np.concatenate(spike_train.firing_times).tolist()
#     sources = [neuron for neuron in network.neurons for _ in range(spike_train.num_spikes(neuron.idx))]
#     indices = np.argsort(firing_times)

#     M = len(firing_times)
#     N = len(firing_times) # to be adapted, can be smaller than M
#     Phi = np.identity(N)
#     A = np.zeros((N, N))
#     A[1:,:-1] = np.identity(N-1)

#     for m in range(M):
#         # current spike is spikes[N + m]
#         for n in range(N):
#             A[0, n] = np.sum(
#                 [
#                     input.weight * input.resp_deriv((firing_times[indices[m]] - input.delay - firing_times[indices[(m - n - 1) % M]]) % spike_train.period) for input in sources[indices[m]].inputs if input.source is sources[indices[(m - n - 1) % M]] 
#                 ]
#             )
#         A[0] /= np.sum(A[0])
#         Phi = A@Phi

#     return Phi


def norm(x:np.ndarray):
    """
    Normalizes the given array to sum to 1.

    Args:
        x (np.ndarray): The array to normalize.

    Returns:
        (np.ndarray): The normalized array.
    """
    return x / np.sum(x)

def mod(x:np.ndarray, modulo:float, offset:float=0.0):
    """
    Returns the componentwise modulo with offset of the given array.

    Args:
        x (np.ndarray): The array to take the modulo of.
        modulo (float): The modulo to use.
        offset (float, optional): The offset. Defaults to 0.0.

    Returns:
        (np.ndarray): The componentwise modulo with offset of the given array.
    """
    return x - np.floor((x - offset) / modulo) * modulo