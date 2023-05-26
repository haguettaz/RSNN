from rsnn.spike_train.distribution import PeriodicSpikeTrainGenerator, SpikeTrainGenerator

spike_train_generator = SpikeTrainGenerator(0.2, 7, 8)


print("SpikeTrain")
spike_train = spike_train_generator.rand(100)
print("num_spikes", spike_train.num_spikes)
print("type", type(spike_train))
print("firing_times", spike_train.firing_times)

print()

print("MultiChannelSpikeTrain")
multi_channel_spike_train = spike_train_generator.rand(100, 10)
print("num_spikes", multi_channel_spike_train.num_spikes)
print("type", type(multi_channel_spike_train))
for idx, spike_train in enumerate(multi_channel_spike_train.spike_trains):
    print(f"firing_times at {idx}", spike_train.firing_times)

print()

periodic_spike_train_generator = PeriodicSpikeTrainGenerator(0.1, 7, 8)

print("PeriodicSpikeTrain")
periodic_spike_train = periodic_spike_train_generator.rand(100)
print("num_spikes", periodic_spike_train.num_spikes)
print("type", type(periodic_spike_train))
print("firing_times", periodic_spike_train.firing_times)

print()

print("MultiChannelPeriodicSpikeTrain")
multi_channel_periodic_spike_train = periodic_spike_train_generator.rand(100, 10)
print("num_spikes", multi_channel_periodic_spike_train.num_spikes)
print("type", type(multi_channel_periodic_spike_train))
for idx, spike_train in enumerate(multi_channel_periodic_spike_train.spike_trains):
    print(f"firing_times at {idx}", spike_train.firing_times)

print()



import numpy as np


def get_phi(multi_channel_periodic_spike_train, network):
    """
    Returns the Phi matrix corresponding to the spike train and the network.

    Args:
        spike_train (MultiChannelPeriodicSpikeTrain): The multi-channel periodic spike train.
        network (Network): The network.

    Returns:
        (np.ndarray): The Phi matrix.
    """
    # firing_times = np.concatenate(spike_train.firing_times).tolist()
    firing_times = np.concatenate([spike_train.firing_times for spike_train in multi_channel_periodic_spike_train.spike_trains])
    sources = [neuron for neuron in network.neurons for _ in range(multi_channel_periodic_spike_train[neuron.idx].num_spikes)]
    indices = np.argsort(firing_times)

    M = len(firing_times)
    N = len(firing_times) # to be adapted, can be smaller than M
    Phi = np.identity(N)
    A = np.zeros((N, N))
    A[1:,:-1] = np.identity(N-1)

    for m in range(M):
        # current spike is spikes[N + m]
        for n in range(N):
            A[0, n] = np.sum(
                [
                    synapse.weight * synapse.response_deriv((firing_times[indices[m]] - synapse.delay - firing_times[indices[(m - n - 1) % M]]) % multi_channel_periodic_spike_train.period) for synapse in sources[indices[m]].synapses if synapse.source is sources[indices[(m - n - 1) % M]] 
                ]
            )
        A[0] /= np.sum(A[0])
        Phi = A@Phi

    return Phi