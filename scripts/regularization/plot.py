import os
import random

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

# from rsnn.network.network import Network
# from rsnn.neuron.neuron_old import Synapse
# from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator

# matplotlib.use("pgf")


plt.style.use("paper")

# from rsnn.network.generator import NetworkGenerator
# from rsnn.network.network import Network
# from rsnn.neuron.neuron import Neuron
# from rsnn.neuron.neuron_old import Synapse
# # from rsnn.spike_train.generator_np import PeriodicSpikeTrainGenerator
# from rsnn.spike_train.periodic_spike_train import PeriodicSpikeTrain
# from rsnn.utils.math import dist_mod
from rsnn.utils.utils import load_object_from_file

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(3.6, 1.4))

for l in trange(100):
    neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}.pkl"))
    axes[0].scatter(
        np.full_like(neuron.weights, l),
        neuron.weights,
        s=0.1,
        alpha=0.05,
        c="C0"
    )

    neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_l1.pkl"))
    axes[1].scatter(
        np.full_like(neuron.weights, l),
        neuron.weights,
        s=0.1,
        alpha=0.05,
        c="C0"
    )

    neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_l2.pkl"))
    axes[2].scatter(
        np.full_like(neuron.weights, l),
        neuron.weights,
        s=0.1,
        alpha=0.05,
        c="C0"
    )

# axes[0].set_xlim(0, 99)
axes[0].set_xlabel("neuron")
# axes[1].set_xlim(0, 99)
axes[1].set_xlabel("neuron")
# axes[2].set_xlim(0, 99)
axes[2].set_xlabel("neuron")

axes[0].set_ylabel("weight")

fig.tight_layout()
fig.savefig("weights_reg.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)