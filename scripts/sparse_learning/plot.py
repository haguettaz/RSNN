import os
import random

import matplotlib
# import cvxpy as cp
# import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from rsnn.network.network import Network
from rsnn.neuron.neuron_old import Synapse
from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator

# matplotlib.use("pgf")


plt.style.use('paper')


NUM_NEURONS = 2000
NOMINAL_THRESHOLD = 1.0
ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

NUM_SYNAPSES = 1000
SYNAPSE_BETA = 5.0
SYNAPSE_DELAY_LIM = (1.0, 60.0)

FIRING_RATE = 0.1

EPS, GAP, SLOPE = 1.0, 1.0, 0.5
WEIGHTS_LIM = (-0.1, 0.1)

network = Network(NUM_NEURONS, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)

list_of_dict = []
for period in range(100, 600, 100):
    for i in trange(200):
        for num_unique_neurons in [NUM_SYNAPSES // 100, NUM_SYNAPSES // 10, NUM_SYNAPSES]:
            if not os.path.exists(f"networks/{period}_{num_unique_neurons}_{i}.pkl"):
                print(f"networks/{period}_{num_unique_neurons}_{i}.pkl does not exist")
                continue
            network.load_from_file(f"networks/{period}_{num_unique_neurons}_{i}.pkl")
            for neuron in network.neurons:
                if neuron.synapses:
                    break
            list_of_dict.append({"feasibility": neuron.opt_status == 2, "\# nonzero synapses": neuron.num_active_synapses, "\# unique neurons": num_unique_neurons, "period": period, "i": i})

df = pd.DataFrame(list_of_dict)
# print(df)
# to continue from here

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.2, 3))

# sns.lineplot(
#     ax=axes[0,
#     data=df, 
#     x="period", 
#     hue="unique neurons", 
#     style="unique neurons",
#     y="feasibility probability", 
#     palette=["C1", "C2", "C3"],
#     markers=True
#     )

sns.lineplot(
    ax=ax,
    data=df[df.feasibility], 
    x="period", 
    hue="\# unique neurons", 
    style="\# unique neurons",
    y="\# nonzero synapses", 
    palette=["C1", "C2", "C3"],
    markers=True
    )

sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.5), ncol=3)

fig.tight_layout()
# plt.show()
fig.savefig('l1_learning.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
