import torch

from rsnn.sim.sim import *
from rsnn.sim.utils import *


def test_compute_drift():
    ref_ftimes = torch.tensor([  1,  22,  43,  67,  88, 111, 132, 157, 178]) 
    sim_ftimes = torch.tensor([-199.0000, -178.0000, -157.0000, -133.0000, -112.0000, -89.0000, -68.0000, -43.0000, -22.0000, 1.0012,  22.0014,  43.0031, 67.0034,  88.0048, 111.0053, 132.0082, 156.0930, 178.0103, 200.9970, 221.9968, 243.0107, 267.0042, 288.0099, 311.0116, 332.0140, 356.0977, 378.0154])
    drift = compute_drift(ref_ftimes, sim_ftimes, 200)
    assert len(drift) == 1
    assert abs(drift[0] + 0.0936) < 1e-4