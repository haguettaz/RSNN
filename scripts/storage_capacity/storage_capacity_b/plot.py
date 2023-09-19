import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from rsnn.utils.utils import load_object_from_file

plt.style.use('paper')


NUM_INPUTS = 500

list_of_dict = []

for period in tqdm(range(100, 1100, 100)): # max period to determine later
    for l in trange(100):
        for num_unique_inputs in [NUM_INPUTS, NUM_INPUTS // 10, NUM_INPUTS // 100]:
            if not os.path.exists(os.path.join("neurons", f"neuron_{l}_{period}_{num_unique_inputs}.pkl")):
                print(os.path.join("neurons", f"neuron_{l}_{period}_{num_unique_inputs}.pkl"), "does not exist")
                continue
            neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_{period}_{num_unique_inputs}.pkl"))
            list_of_dict.append({"feasible": neuron.status == "optimal", "num_unique_inputs": num_unique_inputs, "period": period, "l": l})

df = pd.DataFrame(list_of_dict)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.6, 2.0))

sns.lineplot(
    ax=ax,
    data=df, 
    x="period", 
    hue="num_unique_inputs", 
    style="num_unique_inputs",
    y="feasible", 
    palette=["C0", "C1", "C2"],
    markers=True,
    legend=False
    )

# ax.legend(title="num unique inputs", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("period [ms]")
ax.set_ylabel("feas. prob.")
ax.set_xlim(100, 1000)
ax.set_ylim(0, 1)
fig.tight_layout()
# plt.show()
fig.savefig('storage_capacity_b.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
