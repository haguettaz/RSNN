import random

from .network import Network, Synapse


class NetworkGenerator:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def rand(self, firing_threshold, soma_decay, num_synapses, synapse_decay, synapse_delay_lim, kind=None):
        if firing_threshold <= 0:
            raise ValueError("The firing threshold must be positive.")
        if soma_decay <= 0:
            raise ValueError("The soma decay must be positive.")
        if num_synapses < 0:
            raise ValueError("The number of synapses must be non-negative.")
        if synapse_decay <= 0:
            raise ValueError("The synapse decay must be positive.")
        if synapse_delay_lim[0] < 0:
            raise ValueError("The minimum synapse delay must be non-negative.")
        if synapse_delay_lim[1] < 0:
            raise ValueError("The maximum synapse delay must be non-negative.")
        if synapse_delay_lim[0] > synapse_delay_lim[1]:
            raise ValueError("The minimum synapse delay must be less than the maximum synapse delay.")

        kind = kind or "in"
        if kind not in {"in", "out", "full"}:
            raise ValueError("Invalid kind.")

        network = Network(self.num_neurons, firing_threshold, soma_decay)

        if kind == "in":
            for neuron in network.neurons:
                neuron.synapses = [
                    Synapse(
                        idx,
                        random.choice(network.neurons),
                        random.uniform(*synapse_delay_lim),
                        0,
                        soma_decay,
                        synapse_decay,
                    )
                    for idx in range(num_synapses)
                ]
        elif kind == "out":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return network
