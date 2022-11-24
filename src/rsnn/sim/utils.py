import torch


def compute_drift(ref_ftimes, sim_ftimes, T):
    N_f = len(ref_ftimes)

    drift = []

    if len(sim_ftimes) < N_f:
        return drift

    end_sim_ftimes = sim_ftimes[-N_f:] % T
    
    for _ in range(N_f):
        tmp = (end_sim_ftimes - ref_ftimes) % T
        tmp[tmp > T / 2] -= T
        if ((tmp - tmp.mean()).abs() < 1.0).all():
            # print("ref_ftimes", ref_ftimes)
            # print("sim_ftimes", end_sim_ftimes)
            drift.append(tmp.mean().item())

        ref_ftimes = ref_ftimes.roll(-1)

    return drift