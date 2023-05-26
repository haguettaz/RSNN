import random

from .network import Network, Synapse


class NetworkGenerator:
    """A class for generating random networks of spiking neurons.

    Attributes:
        num_neurons (int): The number of neurons in the network.

    Methods:
        rand(firing_threshold, soma_decay, num_synapses, synapse_decay, synapse_delay_lim, kind=None):
            Generates a random network of spiking neurons with the given parameters.
    """

    def __init__(self, num_neurons):
        """Initializes a NetworkGenerator object with the given number of neurons.

        Args:
            num_neurons (int): The number of neurons in the network.
        """
        self.num_neurons = num_neurons

    def rand(self, firing_threshold, soma_decay, num_synapses, synapse_decay, synapse_delay_lim, kind=None):
        """Generates a random network of spiking neurons with the given parameters.

        Args:
            firing_threshold (float): The threshold at which a neuron fires.
            soma_decay (float): The decay rate of the neuron's soma potential.
            num_synapses (int): The number of synapses per neuron. Not used if `kind` is "full"
            synapse_decay (float): The decay rate of the synapse's weight.
            synapse_delay_lim (tuple): A tuple containing the minimum and maximum delay of any synapse.
            kind (str, optional): The type of network to generate. Can be "in" (constant in-degree), "out" (constant out-degree), or "full" (fully connected). Defaults to "in".

        Returns:
            A Network object representing the generated network.
        """
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
            return network
        
        if kind == "out":
            for source in network.neurons:
                for _ in range(num_synapses):
                    neuron = random.choice(network.neurons)
                    neuron.synapses.append(
                        Synapse(
                            len(neuron.synapses),
                            source,
                            random.uniform(*synapse_delay_lim),
                            0,
                            soma_decay,
                            synapse_decay,
                        )
                    )
            return network

        for neuron in network.neurons:
            neuron.synapses = [
                Synapse(
                    source.idx,
                    source,
                    random.uniform(*synapse_delay_lim),
                    0,
                    soma_decay,
                    synapse_decay,
                )
                for source in network.neurons
            ]

        return network
