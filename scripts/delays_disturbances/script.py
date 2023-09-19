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
    nominal_delays = neuron.delays.copy()

    for std_delays in tqdm(np.logspace(-1, 1, 6)):
        if os.path.exists(os.path.join("neurons_", f"neuron_{l}_l2_{std_delays}.pkl")):
            continue
        
        neuron.delays = np.random.normal(nominal_delays, std_delays)
        input_firing_times = [(ref_firing_times[s][None,:] - np.array([PERIOD, 0])[:,None] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

        neuron.sim(TMAX, DT, firing_times, input_firing_times)
        save_object_to_file(neuron, os.path.join("neurons_", f"neuron_{l}_l2_{std_delays}.pkl"))