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

NUM_SYNAPSES = 500
SYNAPSE_BETA = 5.0
SYNAPSE_DELAY_LIM = (1.0, 60.0)

FIRING_RATE = 0.1

EPS, GAP, SLOPE = 1.0, 1.0, 0.5
WEIGHTS_LIM = (-0.1, 0.1)

network = Network(NUM_NEURONS, NOMINAL_THRESHOLD, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)

list_of_dict = []
for period in range(100, 600, 100):
    for i in trange(5):
        for num_unique_neurons in [NUM_SYNAPSES // 100, NUM_SYNAPSES // 10, NUM_SYNAPSES]:
            if not os.path.exists(f"networks/{period}_{num_unique_neurons}_{i}.pkl"):
                print(f"networks/{period}_{num_unique_neurons}_{i}.pkl does not exist")
                continue
            network.load_from_file(f"networks/{period}_{num_unique_neurons}_{i}.pkl")
            for neuron in network.neurons:
                if neuron.synapses:
                    break
            list_of_dict.append({"feasibility": neuron.opt_status == 2, "num_unique_neurons": num_unique_neurons, "period": period, "i": i})

df = pd.DataFrame(list_of_dict)

# to continue from here

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.6, 2.0))

sns.lineplot(
    ax=ax,
    data=df, 
    x="period", 
    hue="num_unique_neurons", 
    style="num_unique_neurons",
    y="feasibility", 
    palette=["C1", "C2", "C3"],
    markers=True
    )

ax.legend(title="num unique neurons", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("period")
ax.set_ylabel("feas. prob.")
# ax.set_xlim(100, 1200)
fig.tight_layout()
# plt.show()
fig.savefig('feasibility_delays.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
