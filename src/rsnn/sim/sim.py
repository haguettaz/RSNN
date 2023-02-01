
import torch
from torch.utils.cpp_extension import load

sim_cpp = load(name="sim_cpp", sources=[__file__.replace(".py", ".cpp")], verbose=True)

        # gives a list of firing times, the spikes from the outside world
        # the dynamics of some neurons are driven by the spikes
        # for the other, the dynamics are driven by the internal dynamics
        
def associative_recall_sim(tmax, spikes, sources, delays, weights, Tr, beta, theta, etab):
    # gives a list of firing times, the spikes from the outside world
    # the dynamics of all neurons are first driven by the spikes, then by the internal dynamics

    if len(spikes) != sources.size(0):
        raise ValueError("The number of spikes' channels must match the number of neurons.")
    
    return sim_cpp.sim_cpp(0.0, tmax, spikes, sources, delays, weights, Tr, beta, theta, etab)


def spontaneous_recall_sim(tmax, spikes, sources, delays, weights, Tr, beta, theta, etab):
    """_summary_

    Args:
        tmax (_type_): _description_
        sources (_type_): _description_
        delays (_type_): _description_
        weights (_type_): _description_
        Tr (_type_): _description_
        beta (_type_): _description_
        spikes (list of lists): spikes from the outside world, i.e., the list of firing times
        etab (_type_): _description_
    """

    if len(spikes) != sources.size(0):
        raise ValueError("The number of spikes' channels must match the number of neurons.")

    t0 = max(max(s for s in spikes)) # starting time of the autonomous network dynamics
    return sim_cpp.sim_cpp(t0, tmax, spikes, sources, delays, weights, Tr, beta, theta, etab)
    

# def sim(max_t, ref_ftimes, sources, delays, weights, Tr, T, beta, theta, sigma_0, eta_b, L_0, seed):
#     """Run the simulation.

#     Args:
#         max_t (_type_): simulation duration.
#         ref_ftimes (_type_): firing times, reference.
#         sources (_type_): connection sources.
#         delays (_type_): connection delays.
#         weights (_type_): connection weights.
#         Tr (_type_): refractory duration.
#         T (_type_): cycle duration.
#         beta (_type_): impulse response spread.
#         theta (_type_): firing threshold.
#         sigma_0 (_type_): initial noise level.
#         eta_b (_type_): neuron potential noise level.
#         L_0 (_type_): number of sensory neurons, i.e., initializable neurons.
#         seed (_type_): random seed.

#     Returns:
#         _type_: _description_
#     """    
#     L = sources.size(0)
#     sim_ftimes = init_sim_ftimes(ref_ftimes, sigma_0, L_0, T)
#     sim_ftimes = sim_cpp.sim_cpp(
#         max_t,
#         sim_ftimes,
#         sources.int(), 
#         delays.double(), 
#         weights.double(), 
#         Tr, 
#         beta, 
#         theta, 
#         eta_b,
#         seed
#     )
#     return [torch.tensor(sim_ftimes[l]) for l in range(L)]

# def init_sim_ftimes(ref_ftimes, sigma_0, L_0, T):
#     """_summary_

#     Args:
#         ref_ftimes (_type_): firing times, reference.
#         sigma_0 (_type_): initial noise level.
#         L_0 (_type_): number of sensory neurons, i.e., initializable neurons.
#         T (_type_): cycle duration.

#     Returns:
#         _type_: _description_
#     """
#     L = len(ref_ftimes)
#     sim_ftimes = []
#     for l in range(L_0):
#         Nf = ref_ftimes[l].nelement()
#         init_ftimes = ref_ftimes[l] - T + sigma_0 * torch.randn(Nf)
#         sim_ftimes.append(init_ftimes.clamp(None, -1e-6).tolist())

#     for l in range(L_0, L):
#         sim_ftimes.append([])
        
#     return sim_ftimes