import os

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

plt.style.use("paper")

from rsnn.utils.utils import load_object_from_file

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

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.6, 3.6), sharex=True, sharey=True)

# 1. Working fine, slope = 50, firing rate = 10 and period = 200
network = load_object_from_file(os.path.join("networks_a", f"network_{FIRING_RATE}_{SLOPE}.pkl"))

axes[0,0].scatter(
    network.phis.real,
    network.phis.imag,
    s=5,
    # alpha=0.5,
    c="C0"
)

axes[0,0].add_patch(Circle((0, 0), radius=1, facecolor="C2", edgecolor="black", linestyle="--", alpha=0.2))
axes[0,0].set_ylabel("Im")

# 2. Effect of firing rate, slope = 0.5, firing rate = 100 and period = 200
firing_rate = 100
network = load_object_from_file(os.path.join("networks_b", f"network_{firing_rate}_{SLOPE}.pkl"))
axes[0,1].scatter(
    network.phis.real,
    network.phis.imag,
    s=5,
    # alpha=0.5,
    c="C0"
)

axes[0,1].add_patch(Circle((0, 0), radius=1, facecolor="C2", edgecolor="black", linestyle="--", alpha=0.2))

# 3. Effect of slope, slope = 0.2, firing rate = 10 and period = 200
slope = 20
network = load_object_from_file(os.path.join("networks_c", f"network_{FIRING_RATE}_{slope}.pkl"))

axes[1,0].scatter(
    network.phis.real,
    network.phis.imag,
    s=5,
    # alpha=0.5,
    c="C0"
)

axes[1,0].add_patch(Circle((0, 0), radius=1, facecolor="C2", edgecolor="black", linestyle="--", alpha=0.2))
axes[1,0].set_xlabel("Re")
axes[1,0].set_ylabel("Im")

# 4. Effect of one single neuron, slope = 0.5, firing rate = 10 and period = 200
network = load_object_from_file(os.path.join("networks_d", f"network_{FIRING_RATE}_{SLOPE}.pkl"))

axes[1,1].scatter(
    network.phis.real,
    network.phis.imag,
    s=5,
    # alpha=0.5,
    c="C0"
)

axes[1,1].add_patch(Circle((0, 0), radius=1, facecolor="C2", edgecolor="black", linestyle="--", alpha=0.2))
axes[1,1].set_xlabel("Re")

fig.tight_layout()
fig.savefig("temporal_stability.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)