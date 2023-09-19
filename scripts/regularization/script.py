import os

import numpy as np
from tqdm import trange

from rsnn.neuron.neuron import Neuron
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 5000

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_INPUTS = 500

INPUT_BETA = 5.0
INPUT_DELAYS_LIM = (1.0, 60.0)
INPUT_WEIGHTS_LIM = (-0.2, 0.2)

FIRING_RATE = 10 #0.1

EPS, GAP, SLOPE = 1.0, 1.0, 0.5
PERIOD = 200

neuron = Neuron(
    None, NUM_INPUTS, INPUT_BETA, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY
)

REF_FIRING_TIMES = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))

for l in trange(100):
    if os.path.exists(os.path.join("neurons", f"neuron_{l}.pkl")) and os.path.exists(os.path.join("neurons", f"neuron_{l}_l1.pkl")) and os.path.exists(os.path.join("neurons", f"neuron_{l}_l2.pkl")):
        continue

    neuron.idx = l
    neuron.sources = np.random.choice(np.concatenate([np.arange(l), np.arange(l + 1, NUM_NEURONS)]), size=NUM_INPUTS, replace=False)
    neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=NUM_INPUTS)

    firing_times = (REF_FIRING_TIMES[l]).copy()
    input_firing_times = [(REF_FIRING_TIMES[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

    neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE)
    save_object_to_file(neuron, os.path.join("neurons", f"neuron_{l}.pkl"))

    neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE, regularizer="l1")
    save_object_to_file(neuron, os.path.join("neurons", f"neuron_{l}_l1.pkl"))

    neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE, regularizer="l2")
    save_object_to_file(neuron, os.path.join("neurons", f"neuron_{l}_l2.pkl"))