import os

import numpy as np
from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron import Neuron
from rsnn.utils.analysis import get_phis
from rsnn.utils.utils import load_object_from_file, save_object_to_file

NUM_NEURONS = 100

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_INPUTS = 500

PERIOD, FIRING_RATE = 200, 10
INPUT_BETA = 5.0
INPUT_DELAYS_LIM = (1.0, PERIOD)
INPUT_WEIGHTS_LIM = (-0.2, 0.2)

EPS, GAP, SLOPE = 1.0, 1.0, 50

neuron = Neuron(
    None, NUM_INPUTS, INPUT_BETA, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY
)

REF_FIRING_TIMES = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))

# 1. Working fine, slope = 50, firing rate = 10 and period = 200
if not os.path.exists(os.path.join("networks_a", f"network_{FIRING_RATE}_{SLOPE}.pkl")):
    for l in trange(NUM_NEURONS):
        if os.path.exists(os.path.join("neurons_a", f"neuron_{l}_{FIRING_RATE}_{SLOPE}.pkl")):
            continue

        neuron.idx = l
        neuron.sources = np.random.randint(NUM_NEURONS, size=NUM_INPUTS)
        neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=NUM_INPUTS)

        firing_times = (REF_FIRING_TIMES[l]).copy()
        input_firing_times = [(REF_FIRING_TIMES[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

        neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE/100, regularizer="l2")
        assert neuron.status == "optimal"

        save_object_to_file(neuron, os.path.join("neurons_a", f"neuron_{l}_{FIRING_RATE}_{SLOPE}.pkl"))

    neurons = [load_object_from_file(os.path.join("neurons_a", f"neuron_{l}_{FIRING_RATE}_{SLOPE}.pkl")) for l in range(NUM_NEURONS)]
    network = Network(neurons)
    network.phis = get_phis(neurons, REF_FIRING_TIMES[:NUM_NEURONS], PERIOD)

    save_object_to_file(network, os.path.join("networks_a", f"network_{FIRING_RATE}_{SLOPE}.pkl"))

# 2. Effect of firing rate, slope = 0.5, firing rate = 100 and period = 200
firing_rate = 100
ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{firing_rate}.pkl"))
if not os.path.exists(os.path.join("networks_b", f"network_{firing_rate}_{SLOPE}.pkl")):
    for l in trange(NUM_NEURONS):
        if os.path.exists(os.path.join("neurons_b", f"neuron_{l}_{firing_rate}_{SLOPE}.pkl")):
            continue

        neuron.idx = l
        neuron.sources = np.random.randint(NUM_NEURONS, size=NUM_INPUTS)
        neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=NUM_INPUTS)

        firing_times = (ref_firing_times[l]).copy()
        input_firing_times = [(ref_firing_times[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

        neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE/100, regularizer="l2")
        assert neuron.status == "optimal"

        save_object_to_file(neuron, os.path.join("neurons_b", f"neuron_{l}_{firing_rate}_{SLOPE}.pkl"))

    neurons = [load_object_from_file(os.path.join("neurons_b", f"neuron_{l}_{firing_rate}_{SLOPE}.pkl")) for l in range(NUM_NEURONS)]
    network = Network(neurons)
    network.phis = get_phis(neurons, ref_firing_times[:NUM_NEURONS], PERIOD)

    save_object_to_file(network, os.path.join("networks_b", f"network_{firing_rate}_{SLOPE}.pkl"))

# 3. Effect of slope, slope = 0.2, firing rate = 10 and period = 200
slope = 20
if not os.path.exists(os.path.join("networks_c", f"network_{FIRING_RATE}_{slope}.pkl")):
    for l in trange(NUM_NEURONS):
        if os.path.exists(os.path.join("neurons_c", f"neuron_{l}_{FIRING_RATE}_{slope}.pkl")):
            continue

        neuron.idx = l
        neuron.sources = np.random.randint(NUM_NEURONS, size=NUM_INPUTS)
        neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=NUM_INPUTS)

        firing_times = (REF_FIRING_TIMES[l]).copy()
        input_firing_times = [(REF_FIRING_TIMES[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

        neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, slope/100, regularizer="l2")
        assert neuron.status == "optimal"

        save_object_to_file(neuron, os.path.join("neurons_c", f"neuron_{l}_{FIRING_RATE}_{slope}.pkl"))

    neurons = [load_object_from_file(os.path.join("neurons_c", f"neuron_{l}_{FIRING_RATE}_{slope}.pkl")) for l in range(NUM_NEURONS)]
    network = Network(neurons)
    network.phis = get_phis(neurons, REF_FIRING_TIMES[:NUM_NEURONS], PERIOD)

    save_object_to_file(network, os.path.join("networks_c", f"network_{FIRING_RATE}_{slope}.pkl"))

# 4. Effect of one single neuron, slope = 0.5, firing rate = 10 and period = 200
if not os.path.exists(os.path.join("networks_d", f"network_{FIRING_RATE}_{SLOPE}.pkl")):
    neuron.idx = 0
    neuron.sources = np.full(NUM_INPUTS, 0)
    neuron.delays = np.random.uniform(*INPUT_DELAYS_LIM, size=NUM_INPUTS)

    firing_times = (REF_FIRING_TIMES[0]).copy()
    input_firing_times = [(REF_FIRING_TIMES[s] + d).reshape(-1) for s, d in zip(neuron.sources, neuron.delays)]

    neuron.optimize_weights(firing_times, input_firing_times, PERIOD, INPUT_WEIGHTS_LIM, EPS, GAP, SLOPE/100, regularizer="l2")
    assert neuron.status == "optimal"

    save_object_to_file(neuron, os.path.join("neurons_d", f"neuron_0_{FIRING_RATE}_{SLOPE}.pkl"))

    neurons = [load_object_from_file(os.path.join("neurons_d", f"neuron_0_{FIRING_RATE}_{SLOPE}.pkl"))]
    network = Network(neurons)
    network.phis = get_phis(neurons, [REF_FIRING_TIMES[0]], PERIOD)
    
    save_object_to_file(network, os.path.join("networks_d", f"network_{FIRING_RATE}_{SLOPE}.pkl"))