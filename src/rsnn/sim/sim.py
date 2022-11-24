
import torch
from torch.utils.cpp_extension import load

sim_cpp = load(name="sim_cpp", sources=[__file__.replace(".py", ".cpp")], verbose=True)

def sim(max_t, ref_ftimes, sources, delays, weights, Tr, T, beta, theta, sigma, eta, seed):
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
    L = sources.size(0)
    sim_ftimes = init_sim_ftimes(ref_ftimes, sigma, T)
    sim_ftimes = sim_cpp.sim_cpp(
        max_t,
        sim_ftimes,
        sources.int(), 
        delays.double(), 
        weights.double(), 
        Tr, 
        beta, 
        theta, 
        eta,
        seed
    )
    return [torch.tensor(sim_ftimes[l]) for l in range(L)]

def init_sim_ftimes(ref_ftimes, sigma, T):
    """_summary_

    Args:
        spike_sequences (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    L = len(ref_ftimes)
    sim_ftimes = []
    for l in range(L):
        Nf = ref_ftimes[l].nelement()
        sim_ftimes.append(((ref_ftimes[l] + sigma * torch.randn(Nf)) % T - T).tolist())
        
    return sim_ftimes