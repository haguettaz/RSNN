import os

import numpy as np
from tqdm import tqdm, trange

from rsnn.utils.utils import load_object_from_file, save_object_to_file

PERIOD = 200
FIRING_RATE = 10
TMAX, DT = 200.0, 5e-2

REF_FIRING_TIMES = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))

for l in trange(100):
    neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_l2.pkl"))

    firing_times = REF_FIRING_TIMES[neuron.idx] - PERIOD
    input_firing_times = [(REF_FIRING_TIMES[s][None,:] - np.array([PERIOD, 0])[:,None] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

    for nominal_threshold in tqdm(np.arange(0.0, 2.5, 0.5)):
        if os.path.exists(os.path.join("neurons_", f"neuron_{l}_l2_{nominal_threshold}.pkl")):
            continue
        neuron.firing_times = np.copy(firing_times)
        neuron.nominal_threshold = nominal_threshold
        neuron.sim(TMAX, DT, input_firing_times)
        save_object_to_file(neuron, os.path.join("neurons_", f"neuron_{l}_l2_{nominal_threshold}.pkl"))