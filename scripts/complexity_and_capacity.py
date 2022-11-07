from time import time

import pandas as pd
import torch
import torch.multiprocessing as mp

from rsnn.optimization.optimization import compute_weights
from rsnn.optimization.utils import compute_observation_matrices

# from rsnn.simulation.simulation import simulate
# from rsnn.simulation.utils import compute_drift, get_input
from rsnn.spike_sequences.sampling import sample_spike_sequences
from rsnn.spike_sequences.template import segment_spike_sequence

if __name__ == "__main__":

    L = 250
    K = 500

    wb = 0.1
    taub = 60
    theta = 1.0

    Tr = 20
    beta = 2  # spreading hyperparameter
    impulse_resp = lambda t_: (t_ >= 0) * t_ / beta * torch.exp(1 - t_ / beta)
    impulse_resp_deriv = lambda t_: (t_ >= 0) * 1 / beta * (1 - t_ / beta) * torch.exp(1 - t_ / beta)

    delays = torch.FloatTensor(L, K).uniform_(0, taub)
    sources = torch.randint(0, L, (L, K))
    torch.save(delays, f"delays.pt")
    torch.save(sources, f"sources.pt")

    eps = 1  # firing surrounding window
    b = 0.0  # maximum level in silent period
    a = 1.0  # minimum slope in activity period

    gamma_wb = 10.0
    gamma_b = 3.0
    gamma_a = 5.0
    gamma_theta = 10.0

    # config = {
    #     "L": L,
    #     "K": K,
    #     "wb": wb,
    #     "taub": taub,
    #     "theta": theta,
    #     "Tr": Tr,
    #     "beta": beta,
    #     "eps": eps,
    #     "b": b,
    #     "a": a,
    #     "gamma_wb": gamma_wb,
    #     "gamma_b": gamma_b,
    #     "gamma_a": gamma_a,
    #     "gamma_theta": gamma_theta,
    # }

    list_of_config = []

    for alpha in [5, 10, 20, 50, 100]:
        print("alpha = ", alpha, " start...", flush=True)

        T = alpha * Tr
        # config["alpha"] = alpha
        # config["T"] = T

        # generate a random sequence of spikes
        spike_sequences = sample_spike_sequences(L, T, Tr)
        torch.save(spike_sequences, f"spike_sequences_{alpha}.pt")

        observation_matrices = []
        for l in range(L):
            segmentation = segment_spike_sequence(spike_sequences[l], Tr, 1)
            observation_matrices.append(
                compute_observation_matrices(
                    spike_sequences, segmentation, delays[l], sources[l], Tr, impulse_resp, impulse_resp_deriv
                )
            )

        ## OPTIMIZATION
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for l in range(L):
            p = mp.Process(target=compute_weights, args=(l, return_dict, *observation_matrices[l], wb, theta, a, b))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        weights = torch.stack([return_dict[l] for l in range(L)])
        torch.save(weights, f"weights_{alpha}.pt")

        # references = [
        #     (torch.argwhere(spike_sequence).flatten()).tolist() for spike_sequence in torch.unbind(spike_sequences, -1)
        # ]

        # ## SIMULATIONS
        # # without noise
        # config["noise_std"] = 0
        # noisy_references = get_input(references, 0, 0, T)
        # firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
        # config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
        # list_of_config.append(config.copy())
        # print("alpha = ", alpha, " without noise: done!")

        # # low noise
        # config["noise_std"] = Tr / 20
        # for _ in range(5):
        #     noisy_references = get_input(references, 0, Tr / 20, T)
        #     firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
        #     config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
        #     list_of_config.append(config.copy())
        # print("alpha = ", alpha, " with low noise: done!", flush=True)

        # # mid noise
        # config["noise_std"] = Tr / 10
        # for _ in range(5):
        #     noisy_references = get_input(references, 0, Tr / 10, T)
        #     firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
        #     config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
        #     list_of_config.append(config.copy())
        # print("alpha = ", alpha, " with mid noise: done!", flush=True)

        # # high noise
        # config["noise_std"] = Tr / 5
        # for _ in range(5):
        #     noisy_references = get_input(references, 0, Tr / 5, T)
        #     firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
        #     config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
        #     list_of_config.append(config.copy())
        # print("alpha = ", alpha, " with high noise: done!", flush=True)

    # df = pd.DataFrame(list_of_config)
    # df.to_csv("complexity_and_capacity.csv", index=False)
