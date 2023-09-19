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
FIRING_RATE = 10
SIMILARITY_EPS = 0.1
SIMILARITY_KERNEL = lambda x_: (np.abs(x_) < SIMILARITY_EPS) * (SIMILARITY_EPS - np.abs(x_)) / SIMILARITY_EPS

REF_FIRING_TIMES = load_object_from_file(os.path.join("spike_trains", f"firing_times_{PERIOD}_{FIRING_RATE}.pkl"))

list_of_dict = []
for l in trange(100):
    for nominal_threshold in tqdm(np.arange(0.0, 2.5, 0.5)):
        neuron = load_object_from_file(os.path.join("neurons_", f"neuron_{l}_l2_{nominal_threshold}.pkl"))
        precision, recall, _ = single_channel_correlation(REF_FIRING_TIMES[l], neuron.firing_times[neuron.firing_times >= 0], PERIOD, SIMILARITY_KERNEL)
        list_of_dict.append({"neuron": l, "nominal_threshold": nominal_threshold, "similarity": precision, "type": r"$p_\varepsilon(x; \hat{x})$"})
        list_of_dict.append({"neuron": l, "nominal_threshold": nominal_threshold, "similarity": recall, "type": r"$r_\varepsilon(x; \hat{x})$"})


df = pd.DataFrame(list_of_dict)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.6, 1.7))

sns.lineplot(
    ax=ax,
    data=df, 
    x="nominal_threshold", 
    y="similarity", 
    markers=True,
    hue="type",
    style="type",
    palette=["C0", "C1"],
    legend=False
    )

# sns.lineplot(
#     ax=ax,
#     data=df, 
#     x="nominal_threshold", 
#     y="recall", 
#     markers=True,
#     palette=["C2"]
#     )

# ax.legend([r"$p_\varepsilon(x; \hat{x})$", r"$r_\varepsilon(x; \hat{x})$"])
# ax.legend(title="max input delay", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r"$\tilde{\theta}_0$")
ax.set_ylabel("similarity")
# ax.legend(title="")
ax.axvline(1.0, color="black", linestyle="--")
# ax.legend(title="", ncol=2, bbox_to_anchor=(0.5, 1.3), loc="upper center")
# ax.set_xscale("log")
ax.set_xlim(0.0, 2.0)
ax.set_ylim(0, 1)
fig.tight_layout()
# plt.show()
fig.savefig('nominal_threshold_shift.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
