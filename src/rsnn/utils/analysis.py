import numpy as np
from tqdm import trange


def get_phis(neurons, firing_times, period):
#(network, multi_channel_periodic_spike_train):
    ref_firing_times = np.concatenate(firing_times)
    ref_neurons = [neuron for neuron in neurons for _ in firing_times[neuron.idx]] 
    
    indices = np.argsort(ref_firing_times)

    M = len(ref_firing_times)
    # N = len(ref_firing_times)  # to be adapted, can be smaller than M

    Phi = np.identity(M)
    A = np.zeros((M, M))
    A[1:, :-1] = np.identity(M - 1)

    for m in trange(M): # at firing time s_m
        neuron = ref_neurons[indices[m]]

        # current spike is spikes[N + m]
        for n in range(M):
            select = neuron.sources == ref_neurons[indices[(m - n - 1)%M]].idx
            A[0, n] = np.sum(neuron.weights[select] * neuron.input_kernel_prime((ref_firing_times[indices[m]] - ref_firing_times[indices[(m - n - 1)%M]] - neuron.delays[select])%period))

            if neuron.idx == ref_neurons[indices[(m - n - 1)%M]].idx:
                A[0, n] -= neuron.refractory_kernel_prime((ref_firing_times[indices[m]] - ref_firing_times[indices[(m - n - 1)%M]])%period)

        A[0] /= np.sum(A[0])
        Phi = A @ Phi

    # return -np.sort(-np.abs(np.linalg.eigvals(Phi)))
    return np.linalg.eigvals(Phi)
