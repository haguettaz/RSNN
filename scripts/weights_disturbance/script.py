import os

import numpy as np
from tqdm import tqdm, trange

from rsnn.utils.utils import load_object_from_file, save_object_to_file

PERIOD = 200
TMAX, DT = 200.0, 5e-2

ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}.pkl"))

for l in trange(100):
    neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_l2.pkl"))

    firing_times = ref_firing_times[neuron.idx] - PERIOD
    input_firing_times = [(ref_firing_times[s][None,:] - np.array([PERIOD, 0])[:,None] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]
    nominal_weights = neuron.weights.copy()

    for std_weights in tqdm(np.logspace(-2.5, -1, 6)):
        if os.path.exists(os.path.join("neurons_", f"neuron_{l}_l2_{std_weights}.pkl")):
            continue

        neuron.weights = np.random.normal(nominal_weights, std_weights)
        neuron.sim(TMAX, DT, firing_times, input_firing_times)
        save_object_to_file(neuron, os.path.join("neurons_", f"neuron_{l}_l2_{std_weights}.pkl"))