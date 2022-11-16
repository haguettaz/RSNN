import torch


def compute_drift(sim_indices_f, ref_indices_f, tol=1e-2):
    N_f = len(ref_indices_f)
    end_sim_indices_f = sim_indices_f[-N_f:]

    for r in range(N_f):
        ref_indices_f = ref_indices_f.roll(r)
        drifts = end_sim_indices_f - ref_indices_f
        if drifts.var() < tol:
            return drifts.mean()

    return torch.nan