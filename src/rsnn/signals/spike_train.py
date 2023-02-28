from typing import Optional, Union

import numpy as np
from tqdm.autonotebook import tqdm

from .utils import get_phi0, get_spiking_matrix


class SpikeTrain:
    def __init__(self, num_channels, duration, refractory_period, firing_rate=None):
        # 1/firing_rate is the average time it takes for a neuron to a fire after its refractory period
        self.num_channels = num_channels
        self.duration = duration

        if isinstance(refractory_period, (int, float)):
            self.refractory_period = [refractory_period]*self.num_channels
        else:
            self.refractory_period = refractory_period
        assert len(self.refractory_period) == self.num_channels

        if isinstance(firing_rate, float) or firing_rate is None:
            self.firing_rate = [firing_rate]*self.num_channels
        else:
            self.firing_rate = firing_rate
        assert len(self.firing_rate) == self.num_channels

        self.firing_times = [np.array([]) for _ in range(num_channels)]


    def rand(self, res=1.0, rng=None):
        rng = rng or np.random.default_rng()

        self.firing_times = []
        N = int(self.duration / res)
        
        for T_r, lambda_s in zip(self.refractory_period, self.firing_rate):
            lambda_s = lambda_s or 0.5
            
            if lambda_s == 0:
                self.firing_times.append(np.array([]))
                continue
            
            N_r = int(T_r // res)
            q_s = min(lambda_s * res, 1)

            i_z = np.empty(N, dtype=int)
            G = get_spiking_matrix(N_r, q_s) / get_phi0(N_r, q_s) # Rescale G to have largest eigenvalue 1

            # First sample z0 by marginalizing over (z_1, ..., z_{N-1})
            pz = np.linalg.matrix_power(G, N).diagonal()
            if pz.max() == 0:
                raise ValueError(f"Inconsistent combination of parameters: firing_rate = {lambda_s}, refractory_period = {T_r}, duration = {self.duration}!")
            
            i_z[0] = rng.choice(N_r + 1, p=pz / pz.sum())
            
            # Then sample z_1, ..., z_{N-1} by backward filtering forward sampling
            # 1. Backward filtering
            msg_backward = np.zeros((N + 1, N_r + 1))
            msg_backward[N, i_z[0]] = 1.0
            for n in range(N, 0, -1):
                msg_backward[n - 1] = G @ msg_backward[n]

            # 2. Forward sampling
            for n in range(1, N):
                pz = G[i_z[n - 1]] * msg_backward[n]
                i_z[n] = rng.choice(N_r + 1, p=pz / pz.sum())

            self.firing_times.append(np.argwhere(i_z == N_r).flatten().astype(float) * res)
    
    def capacity(self, time_step=1.0):
        N = int(self.duration / time_step)
        C = 1.0

        for T_r, lambda_s in zip(self.refractory_period, self.firing_rate):
            N_r = int(T_r // time_step)
            q_s = min(lambda_s * time_step, 1)
            C *= get_phi0(N_r, q_s)**N

        return C
    
    def entropy(self):
        raise NotImplementedError("Entropy is not implemented yet!")

    def theoretical_activity_rate(self, tol=1e-6):
        activity_rate = []

        for refractory_period, firing_rate in zip(self.refractory_period, self.firing_rate):
            prev_rate = firing_rate / (1 + firing_rate * refractory_period)

            dt = 1
            Nr = refractory_period / dt
            q = firing_rate * dt
            phi0 = get_phi0(Nr, q)
            
            # find the limit rate as dt -> 0
            rate = ((1 - q)**(Nr) * q) / (phi0 ** (Nr + 1) + Nr  * (1 - q) ** Nr * q) / dt
            while abs(rate - prev_rate) > tol:
                dt /= 10
                Nr = refractory_period / dt
                q = firing_rate * dt
                phi0 = get_phi0(Nr, q)
                
                prev_rate = rate
                rate = ((1 - q)**(Nr) * q) / (phi0 ** (Nr + 1) + Nr  * (1 - q) ** Nr * q) / dt

            activity_rate.append(rate)

        return activity_rate
    
    def empirical_activity_rate(self):
        activity_rate = []
        for firing_times in self.firing_times:
            activity_rate.append(firing_times.size / self.duration)
        return activity_rate


# def rand_spike_trains(
#         num_neurons: int,
#         duration: float, 
#         refractory_period: Union[float, list[float]], 
#         time_step: float = 1.0,
#         spiking_probability: Union[float, list[float]] = 0.5, 
#         rng: Optional[np.random.Generator] = None,
#         ):

#     rng = rng or np.random.default_rng()

#     if isinstance(duration, float):
#         length = np.full(num_neurons, duration / time_step, dtype=int)
#     else:
#         length = (np.array(duration) / time_step).astype(int)
    
#     if isinstance(refractory_period, float):
#         refractory_length = np.full(num_neurons, refractory_period // time_step, dtype=int)
#     else:
#         refractory_length = (np.array(refractory_period) // time_step).astype(int)

#     if isinstance(spiking_probability, float):
#         spiking_probability = np.full(num_neurons, spiking_probability)
#     else:
#         spiking_probability = np.array(spiking_probability)

#     spike_trains = []
#     for (N, Nr, p) in tqdm(zip(length, refractory_length, spiking_probability)):
#         if p == 0:
#             spike_trains.append(np.array([]))
#             continue

#         # length = int(duration[l] / time_step)
#         # refractory_length = int(refractory_period[l] / time_step) - 1  

#         i_z = np.empty(N, dtype=int)
#         G = get_spiking_matrix(Nr, p) / get_phi0(Nr, p) # Rescale G to have largest eigenvalue 1

#         # First sample z0 by marginalizing over (z_1, ..., z_{N-1})
#         pz = np.linalg.matrix_power(G, N).diagonal()
#         assert pz.max() > 0
#         i_z[0] = rng.choice(Nr + 1, p=pz / pz.sum())
        
#         # Then sample z_1, ..., z_{N-1} by backward filtering forward sampling
#         # 1. Backward filtering
#         msg_backward = np.zeros((N + 1, Nr + 1))
#         msg_backward[N, i_z[0]] = 1.0
#         for n in range(N, 0, -1):
#             msg_backward[n - 1] = G @ msg_backward[n]

#         # 2. Forward sampling
#         for n in range(1, N):
#             pz = G[i_z[n - 1]] * msg_backward[n]
#             i_z[n] = rng.choice(Nr + 1, p=pz / pz.sum())

#         spike_trains.append(np.argwhere(i_z == Nr).flatten().astype(float) * time_step)

#     if num_neurons == 1:
#         spike_trains = spike_trains[0]

#     return spike_trains

# def cardinality(duration, refractory_period, time_step, approx=True):
#     """
#     Returns the cardinality of the set of periodic firing sequences.

#     Args:
#         N (int): the length.
#         Nr (int): the refractory period.

#     Returns:
#         (int): the cardinality.
#     """

#     N = int(duration / time_step)
#     Nr = int(refractory_period // time_step)
    
#     if approx:
#         return get_phi0(Nr) ** N

#     G = get_spiking_matrix(Nr).astype(np.uint64)
#     return np.linalg.matrix_power(G, N).trace()

# def expected_free_spiking_rate(duration, refractory_period, time_step, lambda_s=0.5):
#     """
#     Returns the expected spiking rate of a neuron with the given parameters.

#     Args:
#         N (int): the length.
#         Nr (int): the refractory period.

#     Returns:
#         (float): the expected spiking rate.
#     """

#     N = int(duration / time_step)
#     Nr = int(refractory_period // time_step)
#     G = get_spiking_matrix(Nr, lambda_s) / get_phi0(Nr, lambda_s)
#     return np.linalg.matrix_power(G, N)[-1,-1] / time_step # already almost normalized
