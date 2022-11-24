import torch

from rsnn.ss.rand import *
from rsnn.ss.utils import *


def test_rand_ss():
    # ss = rand_ss(5000, 10000, 10, 0.8)
    # assert ss.shape == (5000, 10000)
    # assert ss.dtype == torch.bool
    # assert (ss.sum(dim=0) / 5000 - 0.8/(1 + 0.8 * 10)).abs().mean().item() <1e-2 # empirical spiking rate is close to p/(1+p*Nr)
    # assert (ss.sum(dim=1) / (10000 - ss.sum(dim=1) * 10) - 0.8).abs().mean().item() < 1e-2 # empirical spiking probability is close to p

    # assert rand_ss(1, 100, 10, 0).sum() == 0
    # assert rand_ss(1, 100, 9, 1).sum() == 10
    
    ss = rand_ss(10000, 10, 2, 0.5)
    _, counts = torch.unique(ss, dim=0, return_counts=True)
    print(counts)
    


def test_get_spiking_matrix():
    G = get_spiking_matrix(10)
    assert G.shape == (11, 11)
    assert G.dtype == torch.float32
    assert G.sum() == 12

    G = get_spiking_matrix(10, p=0.5)
    assert G.shape == (11, 11)
    assert G.dtype == torch.float32
    assert (G.sum(dim=1) == 1).all()

    G = get_spiking_matrix(10, p=0)
    assert G.shape == (11, 11)
    assert G.dtype == torch.float32
    assert (G.sum(dim=1) == 1).all()

    G = get_spiking_matrix(10, p=1)
    assert G.shape == (11, 11)
    assert G.dtype == torch.float32
    assert (G.sum(dim=1) == 1).all()

def test_get_phi0():
    phi0 = get_phi0(0)
    assert phi0 == 2

    phi0 = get_phi0(1000)
    assert phi0 >= 1 and phi0 < 2

def test_get_cardinality():
    card = get_cardinality(0, 0)
    assert card == 1

    card = get_cardinality(10, 0)
    assert card == 2**10

    card = get_cardinality(50, 10, approx=False)
    assert card == 4676
