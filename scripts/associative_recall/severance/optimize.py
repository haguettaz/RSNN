import os

import numpy as np
from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.utils.analysis import get_phis
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 500

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_INPUTS = 500

PERIOD, FIRING_RATE = 100, 10
INPUT_BETA = 5.0
INPUT_DELAYS_LIM = (1.0, 50.0)
INPUT_WEIGHTS_LIM = (-0.2, 0.2)

EPS, GAP, SLOPE = 1.0, 1.0, 50

neuron = Neuron(
    None, NUM_INPUTS, INPUT_BETA, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY
)

REF_FIRING_TIMES = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))

if not os.path.exists(os.path.join("network", f"network.pkl")):
    for l in trange(NUM_NEURONS):
        if os.path.exists(os.path.join("neurons", f"neuron_{l}.pkl")):
            continue

        neuron.idx = l
        neuron.sources = np.random.randint(NUM_NEURONS, size=NUM_INPUTS)
        neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=NUM_INPUTS)

        firing_times_many = [(REF_FIRING_TIMES[l+i*NUM_NEURONS]).copy() for i in range(3)]
        input_firing_times_many = [[(REF_FIRING_TIMES[s+i*NUM_NEURONS] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)] for i in range(3)]

        neuron.optimize_weights_many(firing_times_many, input_firing_times_many, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE/100, regularizer="l2")
        assert neuron.status == "optimal"

        save_object_to_file(neuron, os.path.join("neurons", f"neuron_{l}.pkl"))

    neurons = [load_object_from_file(os.path.join("neurons", f"neuron_{l}.pkl")) for l in range(NUM_NEURONS)]
    network = Network(neurons)
    # network.phis = get_phis(neurons, REF_FIRING_TIMES[:NUM_NEURONS], PERIOD)

    save_object_to_file(network, os.path.join("network", f"network.pkl"))
