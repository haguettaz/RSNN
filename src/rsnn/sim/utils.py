import torch


def compute_drift(ref_ftimes, sim_ftimes, T, tol=1):
    N_f = len(ref_ftimes)

    if len(sim_ftimes) < N_f:
        return torch.nan

    end_sim_ftimes = sim_ftimes[-N_f:] % T

    print(ref_ftimes, end_sim_ftimes)

    for _ in range(N_f):
        drifts = (end_sim_ftimes - ref_ftimes) % T
        drifts[drifts > T / 2] -= T
        if drifts.var() < tol:
            return drifts.mean()
        ref_ftimes = ref_ftimes.roll(-1)

    return torch.nan