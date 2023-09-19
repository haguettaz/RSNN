import os

import numpy as np

from rsnn.spike_train.measure import multi_channel_correlation, single_channel_correlation
from rsnn.utils.utils import load_object_from_file

NUM_NEURONS = 1000

NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_INPUTS = 500

PERIOD, FIRING_RATE = 200, 10
INPUT_BETA = 5.0
INPUT_DELAYS_LIM = (1.0, PERIOD)
INPUT_WEIGHTS_LIM = (-0.2, 0.2)

EPS, GAP, SLOPE = 1.0, 1.0, 50

REF_FIRING_TIMES = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))

SIMILARITY_EPS = 2.5
SIMILARITY_KERNEL = lambda x_: (np.abs(x_) < SIMILARITY_EPS) * (SIMILARITY_EPS - np.abs(x_)) / SIMILARITY_EPS

print(single_channel_correlation(REF_FIRING_TIMES[0], REF_FIRING_TIMES[0], PERIOD, SIMILARITY_KERNEL))
print(single_channel_correlation(REF_FIRING_TIMES[0], REF_FIRING_TIMES[1], PERIOD, SIMILARITY_KERNEL))

print(multi_channel_correlation(REF_FIRING_TIMES[:10], REF_FIRING_TIMES[:10], PERIOD, SIMILARITY_KERNEL))
print(multi_channel_correlation(REF_FIRING_TIMES[:10], REF_FIRING_TIMES[2:12], PERIOD, SIMILARITY_KERNEL))
