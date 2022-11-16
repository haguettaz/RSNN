
import torch
from torch.utils.cpp_extension import load

sim_cpp = load(name="sim_cpp", sources=[__file__.replace(".py", ".cpp")], verbose=True)

def sim(max_t, spike_sequences, sources, delays, weights, Tr, beta, theta, wb, sigma, eta):
    """_summary_

    Args:
        sources (_type_): _description_
        delays (_type_): _description_
        weights (_type_): _description_
        Tr (_type_): _description_
        beta (_type_): _description_
        theta (_type_): _description_
        spike_sequences (_type_): _description_
        noise_init (_type_): _description_
        noise_potential (_type_): _description_
        max_t (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """
    firing_times = get_initial_firing_times(spike_sequences, sigma)
    sim_firing_times = sim_cpp.sim_cpp(
        max_t,
        firing_times,
        sources.int(), 
        delays.double(), 
        weights.double(), 
        Tr, 
        beta, 
        theta, 
        wb,
        eta,
    )
    return [torch.tensor(sim_firing_times[l]) for l in range(spike_sequences.size(0))]

def get_initial_firing_times(spike_sequences, sigma):
    """_summary_

    Args:
        spike_sequences (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    L, N = spike_sequences.size()
    firing_times = []
    for l in range(L):
        indices_f = torch.argwhere(spike_sequences[l]).flatten()
        firing_times.append(((indices_f + sigma * torch.randn(indices_f.size())) % N - N).tolist())
    
    return firing_times