import torch

from rsnn.sim.sim import *
from rsnn.sim.utils import *


def test_compute_drift():
    ref_ftimes = torch.tensor([0, 2, 5])

    sim_ftimes = torch.tensor([0.6, 2.6, 5.6])
    drift = compute_drift(ref_ftimes, sim_ftimes, 6)
    assert (drift - 0.6).abs() < 1e-6

    sim_ftimes = torch.tensor([0.6, 2.6, 5.6, 6.6, 8.6, 11.6])
    drift = compute_drift(ref_ftimes, sim_ftimes, 6)
    assert (drift - 0.6).abs() < 1e-6

    sim_ftimes = torch.tensor([0.2, 2.8])
    drift = compute_drift(ref_ftimes, sim_ftimes, 6)
    assert drift is torch.nan

    sim_ftimes = torch.tensor([11.997, 14.007, 16.999])
    drift = compute_drift(ref_ftimes, sim_ftimes, 6)
    assert (drift - 1e-3).abs() < 1e-6