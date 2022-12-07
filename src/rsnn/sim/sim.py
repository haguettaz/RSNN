
import torch
from torch.utils.cpp_extension import load

sim_cpp = load(name="sim_cpp", sources=[__file__.replace(".py", ".cpp")], verbose=True)

def sim(max_t, ref_ftimes, sources, delays, weights, Tr, T, beta, theta, sigma_0, sigma_z, seed):
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
    sim_ftimes = init_sim_ftimes(ref_ftimes, sigma_0, T)
    sim_ftimes = sim_cpp.sim_cpp(
        max_t,
        sim_ftimes,
        sources.int(), 
        delays.double(), 
        weights.double(), 
        Tr, 
        beta, 
        theta, 
        sigma_z,
        seed
    )
    return [torch.tensor(sim_ftimes[l]) for l in range(L)]

def init_sim_ftimes(ref_ftimes, sigma_0, T):
    """_summary_

    Args:
        spike_sequences (_type_): _description_
        sigma_0 (_type_): _description_

    Returns:
        _type_: _description_
    """
    L = len(ref_ftimes)
    sim_ftimes = []
    for l in range(L):
        Nf = ref_ftimes[l].nelement()
        init_ftimes = ref_ftimes[l] - T + sigma_0 * torch.randn(Nf)
        sim_ftimes.append(init_ftimes.clamp(None, -1e-6).tolist())
        
    return sim_ftimes