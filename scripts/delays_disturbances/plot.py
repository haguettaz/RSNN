import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from rsnn.spike_train.measure import single_channel_correlation
from rsnn.utils.utils import load_object_from_file

plt.style.use('paper')


PERIOD = 200
SIMILARITY_EPS = 1.0
SIMILARITY_KERNEL = lambda x_: (np.abs(x_) < SIMILARITY_EPS) * (SIMILARITY_EPS - np.abs(x_)) / SIMILARITY_EPS

ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}.pkl"))

list_of_dict = []
for l in trange(100):
    for std_delays in tqdm(np.logspace(-1, 1, 6)):
        neuron = load_object_from_file(os.path.join("neurons_", f"neuron_{l}_l2_{std_delays}.pkl"))
        corr, _ = single_channel_correlation(ref_firing_times[l], neuron.firing_times[neuron.firing_times >= 0], PERIOD, SIMILARITY_KERNEL)
        list_of_dict.append({"neuron": l, "std_delays": std_delays**2, "corr": corr})

df = pd.DataFrame(list_of_dict)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.6, 2.0))

# sns.scatterplot(
#     ax=ax,
#     data=df, 
#     x="std_delays", 
#     y="corr", 
#     legend=False,
#     alpha=0.1,
#     )

sns.lineplot(
    ax=ax,
    data=df, 
    x="std_delays", 
    y="corr", 
    
    markers=True,
    legend=False,
    )

# ax.legend(title="max input delay", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r"$\sigma_{d}^2$")
ax.set_ylabel("similarity")
ax.set_xscale("log")
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(0, 1)
fig.tight_layout()
# plt.show()
fig.savefig('delays_disturbance.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
