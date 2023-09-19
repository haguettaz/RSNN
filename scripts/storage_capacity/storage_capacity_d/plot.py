import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from rsnn.utils.utils import load_object_from_file

plt.style.use('paper')


NUM_INPUTS = 500
NUM_UNIQUE_INPUTS = 5

list_of_dict = []

for period in tqdm(range(100, 400, 50)): # max period to determine later
    for l in trange(100):
        for max_input_delay_div in [4, 2, 1]:
            if not os.path.exists(os.path.join("neurons", f"neuron_{l}_{period}_{max_input_delay_div}.pkl")):
                print(os.path.join("neurons", f"neuron_{l}_{period}_{max_input_delay_div}.pkl"), "does not exist")
                continue
            neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_{period}_{max_input_delay_div}.pkl"))
            list_of_dict.append({"feasible": neuron.status == "optimal", "max_input_delay_div": max_input_delay_div, "period": period, "l": l})

df = pd.DataFrame(list_of_dict)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.6, 2.0))

sns.lineplot(
    ax=ax,
    data=df, 
    x="period", 
    hue="max_input_delay_div", 
    style="max_input_delay_div",
    y="feasible", 
    palette=["C0", "C1", "C2"],
    markers=True,
    legend=False
    )

# ax.legend(title="max input delay", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("period [ms]")
ax.set_ylabel("feas. prob.")
ax.set_xlim(100, 350)
ax.set_ylim(0, 1)
fig.tight_layout()
# plt.show()
fig.savefig('storage_capacity_d.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
