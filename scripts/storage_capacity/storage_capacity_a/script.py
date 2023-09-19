import os

import numpy as np
from tqdm import tqdm, trange

from rsnn.neuron.neuron import Neuron
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 5000

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

INPUT_BETA = 5.0
INPUT_DELAYS_LIM = (1.0, 60.0)
INPUT_WEIGHTS_LIM = (-0.2, 0.2)

FIRING_RATE = 10

EPS, GAP, SLOPE = 1.0, 1.0, 0.5

neuron = Neuron(
    None, 0, INPUT_BETA, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY
)

for period in tqdm(range(100, 1100, 100)):
    ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_{period}_{FIRING_RATE}.pkl"))
    for l in trange(100):
        for num_inputs in [250,500,1000]:
            if os.path.exists(os.path.join("neurons", f"neuron_{l}_{period}_{num_inputs}.pkl")):
                continue
            neuron.idx = l
            neuron.num_inputs = num_inputs
            neuron.sources = np.random.choice(np.concatenate([np.arange(l), np.arange(l + 1, NUM_NEURONS)]), size=num_inputs, replace=False)
            neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=num_inputs)
            firing_times = (ref_firing_times[l]).copy()
            input_firing_times = [(ref_firing_times[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]
            neuron.optimize_weights(firing_times, input_firing_times, period, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE)
            save_object_to_file(neuron, os.path.join("neurons", f"neuron_{l}_{period}_{num_inputs}.pkl"))