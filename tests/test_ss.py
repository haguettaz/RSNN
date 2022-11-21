import torch

from rsnn.ss.rand import *
from rsnn.ss.utils import *


def test_rand_ss():
    ss = rand_ss(20, 100, 10, 0.5)
    assert ss.shape == (20, 100)
    assert ss.dtype == torch.bool

    ss = rand_ss(1, 100000, 10, 0.5)
    assert ss.shape == (1, 100000)
    assert ss.dtype == torch.bool
    assert (ss.sum() / (100000 - ss.sum() * 10) - 0.5 < 0.01)

    ss = rand_ss(1, 100000, 10, 0.8)
    assert ss.shape == (1, 100000)
    assert ss.dtype == torch.bool
    assert (ss.sum() / (100000 - ss.sum() * 10) - 0.8 < 0.01)

    ss = rand_ss(1, 100000, 10, 0.2)
    assert ss.shape == (1, 100000)
    assert ss.dtype == torch.bool
    assert (ss.sum() / (100000 - ss.sum() * 10) - 0.2 < 0.01)

    ss = rand_ss(1, 1000, 10, 0)
    assert ss.shape == (1, 1000)
    assert ss.dtype == torch.bool
    assert ss.sum() == 0

    ss = rand_ss(1, 1000, 9, 1)
    assert ss.shape == (1, 1000)
    assert ss.dtype == torch.bool
    assert ss.sum() == 100

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
