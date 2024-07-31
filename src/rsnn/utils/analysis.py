import numpy as np
from scipy import sparse
from tqdm.autonotebook import trange


def get_phis(neurons, firing_times, period):
#(network, multi_channel_periodic_spike_train):
    flat_neurons = [neuron for neuron in neurons for _ in firing_times[neuron.idx]] 
    flat_firing_times = np.concatenate(firing_times)
    
    indices = np.argsort(flat_firing_times)

    M = len(flat_firing_times)
    # N = len(flat_firing_times)  # TODO: adapt the number of last spikes to consider

    Phi = np.identity(M)
    # A = sparse.diags_array(np.ones(M-1), offsets=-1).tocsc()
    A = sparse.diags(np.ones(M-1), offsets=-1, format="lil")
    r0 = np.zeros(M)
    # A = np.zeros((M, M)) #prefer a sparse representation here
    # A[1:, :-1] = np.identity(M - 1)

    for m in trange(M): # at firing time s_m
        neuron = flat_neurons[indices[m]]

        # current spike is spikes[N + m]
        for n in range(M):
            select = neuron.sources == flat_neurons[indices[(m - n - 1)%M]].idx
            r0[n] = np.sum(neuron.weights[select] * neuron.input_kernel_prime((flat_firing_times[indices[m]] - flat_firing_times[indices[(m - n - 1)%M]] - neuron.delays[select])%period))
            # A[0, n] = np.sum(neuron.weights[select] * neuron.input_kernel_prime((flat_firing_times[indices[m]] - flat_firing_times[indices[(m - n - 1)%M]] - neuron.delays[select])%period))
            # if neuron.idx == flat_neurons[indices[(m - n - 1)%M]].idx:
            #     A[0, n] -= neuron.refractory_kernel_prime((flat_firing_times[indices[m]] - flat_firing_times[indices[(m - n - 1)%M]])%period)

        A[0] = r0 / np.sum(r0)
        Phi = A @ Phi

    # return -np.sort(-np.abs(np.linalg.eigvals(Phi)))
    return np.linalg.eigvals(Phi)
