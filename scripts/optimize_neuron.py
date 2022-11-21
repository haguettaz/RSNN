import argparse

import torch

from rsnn.optim.optim import compute_weights
from rsnn.optim.utils import compute_observation_matrices

# from rsnn.simulation.simulation import simulate
# from rsnn.simulation.utils import compute_drift, get_input
from rsnn.ss.rand import sample_spike_sequences
from rsnn.ss.template import segment_spike_sequence

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optimize over the weights of a spiking neuron.")
    parser.add_argument("--num_neuron", "-l", type=int, help="the neuron to optimize")
    parser.add_argument("--num_neurons", "-L", type=int, help="the number of neurons")
    parser.add_argument("--num_synapses", "-K", type=int, help="the number of synapses per neuron", default=500)
    parser.add_argument(
        "--alpha", type=int, help="the duration of the firing sequences in refractroy period", default=10
    )
    args = parser.parse_args()

    wb = 0.1
    taub = 60
    theta = 1.0

    Tr = 20
    beta = 1.8  # spreading hyperparameter
    impulse_resp = lambda t_: (t_ >= 0) * t_ / beta * torch.exp(1 - t_ / beta)
    impulse_resp_deriv = lambda t_: (t_ >= 0) * 1 / beta * (1 - t_ / beta) * torch.exp(1 - t_ / beta)

    torch.manual_seed(42)
    delays = torch.load("delays.pt")
    sources = torch.load("sources.pt")

    if delays.shape[0] != args.num_neurons or delays.shape[1] != args.num_synapses:
        raise ValueError("The number of neurons or synapses does not match the saved delays.")

    if sources.shape[0] != args.num_neurons or sources.shape[1] != args.num_synapses:
        raise ValueError("The number of neurons or synapses does not match the saved sources.")

    eps = 1  # firing surrounding window
    b = 0.0  # maximum level in silent period
    a = 1.0  # minimum slope in activity period

    gamma_wb = 10
    gamma_theta = 10
    gamma_a = 5
    gamma_b = 3

    T = args.alpha * Tr

    # generate a random sequence of spikes
    spike_sequences = torch.load(f"spike_sequences_{args.alpha}.pt")

    if spike_sequences.shape[0] != args.num_neurons or spike_sequences.shape[1] != T:
        raise ValueError(
            "The number of neurons or the duration of the spike sequences does not match the requirements."
        )

    # compute the observation matrices and optimize for the weights
    segmentation = segment_spike_sequence(spike_sequences[args.num_neuron], Tr, 1)
    C_theta, C_a, C_b = compute_observation_matrices(
        spike_sequences,
        segmentation,
        delays[args.num_neuron],
        sources[args.num_neuron],
        Tr,
        impulse_resp,
        impulse_resp_deriv,
    )
    weights = compute_weights(C_theta, C_a, C_b, wb, theta, a, b, gamma_wb, gamma_theta, gamma_a, gamma_b)
    print("Neuron optimization is done!", flush=True)

    torch.save(weights, f"weights_{args.alpha}_{args.num_neuron}.pt")

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
