import numpy as np
from matplotlib import pyplot as plt

from rsnn.spike_train.generator import PeriodicSpikeTrainGenerator

plt.style.use("paper")

from rsnn.utils.math import dist_mod

NOMINAL_THRESHOLD = 1.0

ABSOLUTE_REFRACTORY = 5.0
RELATIVE_REFRACTORY = 5.0

REFRACTORY_KERNEL = lambda t_: np.select(
    [t_ > ABSOLUTE_REFRACTORY, t_ > 0], [np.exp(-(t_ - ABSOLUTE_REFRACTORY) / RELATIVE_REFRACTORY), np.inf], 0
)

PERIOD = 100
FIRING_RATE = 0.1

EPS, GAP, SLOPE = 1.0, 1.0, 0.5

spike_train_generator = PeriodicSpikeTrainGenerator(FIRING_RATE, ABSOLUTE_REFRACTORY, RELATIVE_REFRACTORY)
spike_train = spike_train_generator.rand(PERIOD)

firing_threshold = lambda t_: NOMINAL_THRESHOLD + np.sum(
    REFRACTORY_KERNEL((t_[:, None] - spike_train.firing_times[None, :]) % PERIOD), axis=-1
)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.6, 2.4))

times = np.linspace(0, PERIOD, 1000)
time_dist = dist_mod(times[None, :], spike_train.firing_times[:, None], PERIOD)
mask = np.any(time_dist < EPS, axis=0)

axes[0].plot(times, firing_threshold(times), c="black", alpha=0.7)
for s_ in spike_train.firing_times:
    axes[0].axvline(s_, c="black", linestyle="--", alpha=0.5)
    axes[1].axvline(s_, c="black", linestyle="--", alpha=0.5)

# equal
axes[0].scatter(
    spike_train.firing_times,
    firing_threshold(spike_train.firing_times),
    marker="x",
    c="C2",
)

# upper bound
neuron_potential_ub = firing_threshold(times) - GAP
neuron_potential_ub[mask] += GAP
axes[0].fill_between(times, neuron_potential_ub, 3.0, alpha=0.2, color="C1")

# lower bound
neuron_potential_prime_lb = np.full_like(times, -np.inf)
neuron_potential_prime_lb[mask] = SLOPE
axes[1].fill_between(times, neuron_potential_prime_lb, -1.0, alpha=0.2, color="C1")

axes[0].set_ylim(-0.5, 2.5)
axes[1].set_ylim(-0.5, 2.5)
axes[0].set_xlim(0, PERIOD)
axes[0].set_ylabel("$z$-template")
axes[1].set_ylabel("$z'$-template")
axes[1].set_xlabel("time [ms]")

fig.tight_layout()
fig.savefig("template.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
