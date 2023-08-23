import math

import numpy as np


def input_spike_resp(synapse_decay, soma_decay, t):
    tmax = (math.log(synapse_decay) - math.log(soma_decay)) / (1 / soma_decay - 1 / synapse_decay)
    gamma = 1 / (math.exp(-tmax / soma_decay) - math.exp(-tmax / synapse_decay))

    if isinstance(t, np.ndarray):
        z = gamma * (np.exp(-t / soma_decay) - np.exp(-t / synapse_decay))
        z[t < 0] = 0.0
        return z
    
    if t < 0:
        return 0.0
    
    return gamma * (np.exp(-t / soma_decay) - np.exp(-t / synapse_decay))

def input_spike_resp_deriv(synapse_decay, soma_decay, t):
    tmax = (math.log(synapse_decay) - math.log(soma_decay)) / (1 / soma_decay - 1 / synapse_decay)
    gamma = 1 / (math.exp(-tmax / soma_decay) - math.exp(-tmax / synapse_decay))

    if isinstance(t, np.ndarray):
        z = gamma * (np.exp(-t / synapse_decay) / synapse_decay - np.exp(-t / soma_decay) / soma_decay)
        z[t < 0] = 0.0
        return z
    
    if t < 0:
        return 0.0
    
    return gamma * (np.exp(-t / synapse_decay) / synapse_decay - np.exp(-t / soma_decay) / soma_decay)

def refractory_spike_resp(rel_refractory_period, abs_refractory_period, refractory_weight, t):
    if isinstance(t, np.ndarray):
        z = -refractory_weight * np.exp(- (t - abs_refractory_period) / rel_refractory_period)
        z[t <= abs_refractory_period] = -np.inf
        z[t <= 0] = 0.0
        return z
    
    if t <= 0:
        return 0.0
    
    if t <= abs_refractory_period:
        return -np.inf
    
    return -refractory_weight * np.exp(- (t - abs_refractory_period) / rel_refractory_period)