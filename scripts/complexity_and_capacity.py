from time import time

import pandas as pd
import torch

from rsnn.memorization.posterior import compute_posterior
from rsnn.memorization.prior import compute_box_prior
from rsnn.memorization.template import parse_spike_sequences
from rsnn.memorization.utils import compute_observation_matrices
from rsnn.simulation.simulation import simulate
from rsnn.simulation.utils import compute_drift, get_input
from rsnn.spike_sequences.sampling import sample_spike_sequences

if __name__ == "__main__":

    L = 2
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
    weights = torch.FloatTensor(L, K).uniform_(-wb, wb)

    eps = 1  # firing surrounding window
    b = 0.0  # maximum level in silent period
    a = 1.0  # minimum slope in activity period

    gamma_wb = 10.0
    gamma_b = 3.0
    gamma_a = 5.0
    gamma_theta = 10.0

    config = {
        "L": L,
        "K": K,
        "wb": wb,
        "taub": taub,
        "theta": theta,
        "Tr": Tr,
        "beta": beta,
        "eps": eps,
        "b": b,
        "a": a,
        "gamma_wb": gamma_wb,
        "gamma_b": gamma_b,
        "gamma_a": gamma_a,
        "gamma_theta": gamma_theta,
    }

    list_of_config = []

    for alpha in [5, 10, 20, 50, 100]:
        print("alpha = ", alpha, " start...", flush=True)

        T = alpha * Tr
        config["alpha"] = alpha
        config["T"] = T

        # generate a random sequence of spikes
        spike_sequences = sample_spike_sequences(T, Tr, L)

        firing_indices, active_period_indices, silent_period_indices = parse_spike_sequences(spike_sequences, Tr, eps)
        C = compute_observation_matrices(spike_sequences, delays, sources, Tr, impulse_resp, impulse_resp_deriv)

        ## OPTIMIZATION
        mw_f = weights
        Vw_f = torch.ones(L, K)

        mz_b = torch.zeros(L, T, 2)
        Vz_b = torch.ones(L, T, 2)

        mz_b[silent_period_indices[:, 0], silent_period_indices[:, 1], 0] = b
        mz_b[active_period_indices[:, 0], active_period_indices[:, 1], 1] = a
        mz_b[firing_indices[:, 0], firing_indices[:, 1], 0] = theta

        for _ in range(200):
            # posteriors
            mw, Vw, mz = compute_posterior(mw_f, Vw_f, mz_b, Vz_b, C)

            # priors
            mw_f, Vw_f = compute_box_prior(mw, -wb, wb, gamma_wb)

            mz_b = torch.zeros(T, L, 2)
            Vz_b = 1e9 * torch.ones(T, L, 2)
            (
                mz_b[silent_period_indices[:, 0], silent_period_indices[:, 1], 0],
                Vz_b[silent_period_indices[:, 0], silent_period_indices[:, 1], 0],
            ) = compute_box_prior(mz[silent_period_indices[:, 0], silent_period_indices[:, 1], 0], None, b, gamma_b)
            (
                mz_b[active_period_indices[:, 0], active_period_indices[:, 1], 1],
                Vz_b[active_period_indices[:, 0], active_period_indices[:, 1], 1],
            ) = compute_box_prior(mz[active_period_indices[:, 0], active_period_indices[:, 1], 1], a, None, gamma_a)
            (
                mz_b[firing_indices[:, 0], firing_indices[:, 1], 0],
                Vz_b[firing_indices[:, 0], firing_indices[:, 1], 0],
            ) = compute_box_prior(mz[firing_indices[:, 0], firing_indices[:, 1], 0], theta, theta, gamma_theta)
            print("alpha = ", alpha, " optimization done in ", time() - t0, "s", flush=True)

        # optimization loss results
        config["w_err"] = ((mw - wb).abs() + (mw + wb).abs() - 2 * wb).sum()
        config["theta_err"] = (mz[firing_indices[:, 0], firing_indices[:, 1], 0] - theta).abs().sum()
        config["a_err"] = (
            (mz[active_period_indices[:, 0], active_period_indices[:, 1], 1] - a).abs()
            + (mz[active_period_indices[:, 0], active_period_indices[:, 1], 1] - 1e9).abs()
            - (1e9 - a)
        ).sum()
        config["b_err"] = (
            (mz[active_period_indices[:, 0], active_period_indices[:, 1], 0] - b).abs()
            + (mz[active_period_indices[:, 0], active_period_indices[:, 1], 1] + 1e9).abs()
            - (b + 1e9)
        ).sum()

        references = [
            (torch.argwhere(spike_sequence).flatten()).tolist() for spike_sequence in torch.unbind(spike_sequences, -1)
        ]

        ## SIMULATIONS
        # without noise
        config["noise_std"] = 0
        noisy_references = get_input(references, 0, 0, T)
        firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
        config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
        list_of_config.append(config.copy())
        print("alpha = ", alpha, " without noise: done!")

        # low noise
        config["noise_std"] = Tr / 20
        for _ in range(5):
            noisy_references = get_input(references, 0, Tr / 20, T)
            firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
            config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
            list_of_config.append(config.copy())
        print("alpha = ", alpha, " with low noise: done!", flush=True)

        # mid noise
        config["noise_std"] = Tr / 10
        for _ in range(5):
            noisy_references = get_input(references, 0, Tr / 10, T)
            firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
            config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
            list_of_config.append(config.copy())
        print("alpha = ", alpha, " with mid noise: done!", flush=True)

        # high noise
        config["noise_std"] = Tr / 5
        for _ in range(5):
            noisy_references = get_input(references, 0, Tr / 5, T)
            firing_times = simulate(sources, delays, mw, theta, Tr, impulse_resp, noisy_references, 10 * T, 0.001)
            config["drift_mean"], config["drift_std"] = compute_drift(firing_times, references, T, 9)
            list_of_config.append(config.copy())
        print("alpha = ", alpha, " with high noise: done!", flush=True)

    df = pd.DataFrame(list_of_config)
    df.to_csv("complexity_and_capacity.csv", index=False)
