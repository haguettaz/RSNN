import numpy as np
from tqdm.autonotebook import tqdm


def get_stability_matrix(network, spike_train):
    # assumptions: 
    # - causal network
    # - a spike older than spike_train.duration does not influence (directly) the network anymore

    def get_jitter_influence(neuron, ft, neuron_src, ft_src):
        res = 0.0
        if neuron == neuron_src:
            res += neuron.refractory_resp_deriv((ft - ft_src) % spike_train.duration) # should take into account the refractoriness...
        
        for k in range(neuron.num_synapses):
            if neuron.sources[k] == neuron_src:
                res += neuron.weights[k] * neuron.impulse_resp_deriv((ft - ft_src - neuron.delays[k]) % spike_train.duration)

        return res

    def get_index(neuron_idx, firing_time):
        return np.argwhere(spike_train.firing_times[neuron_idx] % spike_train.duration == firing_time) + cum_num_firing_times[neuron_idx]

    cum_num_firing_times = np.cumsum([0] + [spike_train.num_spikes(neuron.idx) for neuron in network.neurons])
    firing_times = np.unique(np.concatenate(spike_train.firing_times))
    dict_firing_times = {}
    for ft in firing_times:
        dict_firing_times[ft] = [neuron for neuron in network.neurons if ft in spike_train.firing_times[neuron.idx]]

    Phi = np.identity(cum_num_firing_times[-1])
    for ft in tqdm(firing_times):
        A = np.identity(cum_num_firing_times[-1])
        for neuron in dict_firing_times[ft]:
            for neuron_src in network.neurons:
                firing_times_src = spike_train.firing_times[neuron_src.idx]
                firing_times_src[firing_times_src >= ft] -= spike_train.duration
                for ft_src in spike_train.firing_times[neuron_src.idx]:
                    A[get_index(neuron.idx, ft), get_index(neuron_src.idx, ft_src)] = get_jitter_influence(neuron, ft, neuron_src, ft_src)
        A /= A.sum(axis=1, keepdims=True)
        Phi = A @ Phi

    return Phi