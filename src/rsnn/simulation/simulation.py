import math

import torch
from tqdm.auto import tqdm


# def simulate(sources, delays, weights, theta, Tr, impulse_resp, firing_times, duration, time_step):
#     """Simulates the network dynamics.

#     Args:
#         sources (torch.LongTensor):
#         delays (torch.FloatTensor): _description_
#         weights (torch.FloatTensor): _description_
#         theta (float): _description_
#         Tr (int or float): _description_
#         impulse_resp (callable): _description_
#         firing_times (list): list of neuron list of firing times. Firing times are assumed to be in increasing order for each neuron.
#         duration (int or float): _description_
#         time_step (float): _description_

#     Returns:
#         (list): list of firing times for each neuron.

#     Raises:
#         ValueError: _description_
#     """
#     L = sources.size(0)

#     M = math.ceil(delays.max() / Tr) + 1

#     active_firing_times = -1e9 * torch.ones(L, M, dtype=torch.float)
#     for l in range(L):
#         num_firings = min(len(firing_times[l]), M)
#         active_firing_times[l, -num_firings:] = torch.tensor(firing_times[l][-num_firings:])

#     if active_firing_times.max() >= 0:
#         raise ValueError("The initial state cannot contain spikes in the future...")

#     pbar = tqdm(desc="Simulating", total=duration, leave=False)
#     time = 0
#     last_firing = 0
#     while time < duration:
#         active_neurons = torch.argwhere(time - active_firing_times[:, -1] > Tr).flatten()
#         neuron_potential = (
#             impulse_resp(time - active_firing_times[sources[active_neurons]] - delays[active_neurons, :, None]).sum(
#                 dim=-1
#             )
#             * weights[active_neurons]
#         ).sum(dim=-1)
#         firing_neurons = active_neurons[neuron_potential >= theta]

#         active_firing_times[firing_neurons, :-1] = active_firing_times[firing_neurons, 1:]
#         active_firing_times[firing_neurons, -1] = time

#         for l in firing_neurons.tolist():
#             firing_times[l].append(time)
#             last_firing = time

#         if time - last_firing > M * Tr:
#             # stop the simulation if no more firing in the network receptive field
#             print("The network is dead...")
#             return firing_times

#         time += time_step
#         pbar.update(time_step)
#     pbar.close()

#     return firing_times
