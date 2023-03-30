
    # def get_stability_matrices(self, tol=1e-6):
    #     for neuron in self.neurons:
    #         neuron.set_num_active_spikes(tol)
        
    #     # set matrix dimension and indices
    #     dim = np.sum([neuron.num_active_spikes for neuron in self.neurons])
    #     indices = np.cumsum([0] + [neuron.num_active_spikes for neuron in self.neurons])

    #     # one stability matrix for each firing pattern
    #     stability_matrices = []
    #     for i in range(self.num_firing_patterns):
    #         # extract all firing times of the network (plus the associated neurons) and sort them by time
    #         firing_times = []
    #         firing_neurons = []
    #         for neuron in self.neurons:
    #             firing_times.append(neuron.firing_patterns[i])
    #             firing_neurons.append(np.full_like(neuron.firing_patterns[i], neuron.idx))
    #         firing_times = np.concatenate(firing_times)
    #         firing_neurons = np.concatenate(firing_neurons).astype(int)
            
    #         Phi = np.identity(dim)
    #         for t in np.unique(firing_times):
    #             # sorted according to indices
    #             mask = (firing_times == t)

    #             firing_neurons_indices = indices[firing_neurons[mask]]

    #             A = np.identity(dim)
    #             # for neurons that fire at time t
    #             for idx in firing_neurons_indices:
    #                 neuron = self.get_neuron(idx)
    #                 # compute influence of all active spikes of the sources on the new last spike
    #                 for source in set(neuron.sources):
    #                     for j, s in enumerate(source.active_spikes):
    #                         A[idx, indices[source.idx] + j] = neuron.spike_influence_deriv(t, source, s)
                    
    #                 # shift all last spikes of the neuron by one
    #                 for j in range(1, neuron.num_active_spikes):
    #                     A[idx + j] = np.roll(A[idx + j], 1)

    #             Phi = A @ Phi

    #         stability_matrices.append(Phi)
    #     return stability_matrices