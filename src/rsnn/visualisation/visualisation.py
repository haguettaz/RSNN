import numpy as np


def add_spike_sequences(ax, firing_times, T, duration, references):
    ax.set_xlim(0, T)
    ax.set_xticks(np.linspace(0, T, 11))
    ax.set_xticklabels(np.linspace(0, 1, 11).round(1))

    qmin = -(duration // T) + 1
    qmax = 1
    ax.set_ylim(qmin, qmax + 1)

    ax.set_yticks(range(qmin, qmax + 1))
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    # ax.set_ylabel("$x_\ell$", labelpad=15, rotation=0, fontsize=12)

    dynamics_handle = None

    for s in references:
        ref_handle = ax.axvline(s % T, color="C3", label="reference")

    for s in firing_times:
        q, r = divmod(s, T)
        if q < 0:
            init_handle = ax.stem(
                r, -q + 0.7, bottom=-q, basefmt=" ", linefmt="C0-", markerfmt="C0.", label="initial state"
            )
        else:
            dynamics_handle = ax.stem(
                r, -q + 0.7, bottom=-q, basefmt=" ", linefmt="C1-", markerfmt="C1.", label="dynamics"
            )

    ax.legend(
        handles=[ref_handle, init_handle] if dynamics_handle is None else [ref_handle, init_handle, dynamics_handle],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    return ax


def add_background(ax, firing_times, T, Tr, eps):
    silent_handle = ax.axvspan(0, T, facecolor="C0", alpha=0.3, label="silent period")

    for sn in firing_times:
        if sn - eps < 0:
            # ax.axvspan(sn - eps + T, sn + eps, facecolor="C1", alpha=0.3)
            ax.axvspan((sn - eps) % T, T, facecolor="white")
            ax.axvspan(0, sn + Tr, facecolor="white")
            active_handle = ax.axvspan((sn - eps) % T, T, facecolor="C3", alpha=0.3, label="active period")
            active_handle = ax.axvspan(0, sn + eps, facecolor="C3", alpha=0.3, label="active period")
            refratory_handle = ax.axvspan(sn + eps, sn + Tr, facecolor="C2", alpha=0.3, label="refractory period")

        elif sn + eps > T - 1:
            ax.axvspan(0, (sn + Tr) % T, facecolor="white")
            ax.axvspan(sn - eps, T, facecolor="white")
            active_handle = ax.axvspan((sn - eps), T, facecolor="C3", alpha=0.3, label="active period")
            active_handle = ax.axvspan(0, (sn + eps) % T, facecolor="C3", alpha=0.3, label="active period")
            refratory_handle = ax.axvspan(
                (sn + eps) % T, (sn + Tr) % T, facecolor="C2", alpha=0.3, label="refractory period"
            )

        elif sn + Tr > T - 1:
            ax.axvspan(0, (sn + Tr) % T, facecolor="white")
            ax.axvspan(sn - eps, T, facecolor="white")
            active_handle = ax.axvspan((sn - eps), sn + eps, facecolor="C3", alpha=0.3, label="active period")
            refratory_handle = ax.axvspan(sn + eps, T, facecolor="C2", alpha=0.3, label="refractory period")
            refratory_handle = ax.axvspan(0, (sn + Tr) % T, facecolor="C2", alpha=0.3, label="refractory period")

        else:
            ax.axvspan(sn - eps, sn + Tr, facecolor="white")
            active_handle = ax.axvspan(sn - eps, sn + eps, facecolor="C3", alpha=0.3, label="active period")
            refratory_handle = ax.axvspan(sn + eps, sn + Tr, facecolor="C2", alpha=0.3, label="refractory period")

        firing_handle = ax.axvline(sn, color="C3", linewidth=0.8, label="firing times")

    handles = [firing_handle, active_handle, silent_handle, refratory_handle]
    return ax, handles
