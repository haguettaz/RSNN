import numpy as np


def get_phi1(network, multi_channel_periodic_spike_train):
    firing_times = np.concatenate([spike_train.firing_times for spike_train in multi_channel_periodic_spike_train.spike_trains])  # type: ignore
    sources = [neuron for neuron in network.neurons for _ in range(multi_channel_periodic_spike_train[neuron.idx].num_spikes)]  # type: ignore
    indices = np.argsort(firing_times)

    M = len(firing_times)
    N = len(firing_times)  # to be adapted, can be smaller than M
    Phi = np.identity(N)
    A = np.zeros((N, N))
    A[1:, :-1] = np.identity(N - 1)

    for m in range(M):
        # current spike is spikes[N + m]
        for n in range(N):
            A[0, n] = np.sum(
                [
                    synapse.weight
                    * synapse.response_deriv(
                        (firing_times[indices[m]] - synapse.delay - firing_times[indices[(m - n - 1) % M]])
                        % multi_channel_periodic_spike_train.period
                    )
                    for synapse in sources[indices[m]].synapses
                    if synapse.source is sources[indices[(m - n - 1) % M]]
                ]
            )
        A[0] /= np.sum(A[0])
        Phi = A @ Phi

    eigvals = np.linalg.eigvals(Phi)
    sorted_mod_eigvals = -np.sort(-np.abs(eigvals))
    return sorted_mod_eigvals[1] if len(sorted_mod_eigvals) > 1 else np.nan
