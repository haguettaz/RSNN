import os

import numpy as np
from tqdm import tqdm

from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator
from rsnn.utils.utils import save_object_to_file

NUM_NEURONS = 5000
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

for firing_rate in tqdm([0.1, 0.2, 0.5, 1.0, 2.0]):
    spike_train_generator = PeriodicSpikeTrainGenerator(firing_rate, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)
    if os.path.exists(os.path.join("spike_trains", f"firing_times_200_{int(firing_rate*100)}.pkl")):
        continue
    firing_times = spike_train_generator.rand(200, NUM_NEURONS)  
    for fts in firing_times:
        assert np.all((fts[1:] - fts[:-1])%200 > ABSOLUTE_REFRACTORY)
    save_object_to_file(firing_times, os.path.join("spike_trains", f"firing_times_200_{int(firing_rate*100)}.pkl"))
    
for period in tqdm(range(100, 400, 50)):
    spike_train_generator = PeriodicSpikeTrainGenerator(0.1, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)
    if os.path.exists(os.path.join("spike_trains", f"firing_times_{period}_10.pkl")):
        continue
    firing_times = spike_train_generator.rand(period, NUM_NEURONS) 
    for fts in firing_times:
        assert np.all((fts[1:] - fts[:-1])%period > ABSOLUTE_REFRACTORY)
    save_object_to_file(firing_times, os.path.join("spike_trains", f"firing_times_{period}_10.pkl"))
    
for period in tqdm(range(400, 1100, 100)):
    spike_train_generator = PeriodicSpikeTrainGenerator(0.1, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)
    if os.path.exists(os.path.join("spike_trains", f"firing_times_{period}_10.pkl")):
        continue
    firing_times = spike_train_generator.rand(period, NUM_NEURONS)  
    for fts in firing_times:
        assert np.all((fts[1:] - fts[:-1])%period > ABSOLUTE_REFRACTORY)
    save_object_to_file(firing_times, os.path.join("spike_trains", f"firing_times_{period}_10.pkl"))
