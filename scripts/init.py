import argparse

import torch

from rsnn.ss.rand import sample_spike_sequences

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optimize over the weights of a recurrent neural network.")
    parser.add_argument("--num_neurons", "-L", type=int, help="the number of neurons", default=250)
    parser.add_argument("--num_synapses", "-K", type=int, help="the number of synapses per neuron", default=500)
    parser.add_argument(
        "--alpha", type=int, help="the duration of the firing sequences in refractroy period", default=10, nargs="+"
    )
    args = parser.parse_args()

    Tr = 20
    beta = 1.8  # spreading hyperparameter
    impulse_resp = lambda t_: (t_ >= 0) * t_ / beta * torch.exp(1 - t_ / beta)
    impulse_resp_deriv = lambda t_: (t_ >= 0) * 1 / beta * (1 - t_ / beta) * torch.exp(1 - t_ / beta)

    wb = 0.1
    taub = 3*Tr
    theta = 1.0

    torch.manual_seed(42)
    delays = torch.FloatTensor(args.num_neurons, args.num_synapses).uniform_(0, taub)
    sources = torch.randint(0, args.num_neurons, (args.num_neurons, args.num_synapses))

    torch.save(delays, f"delays.pt")
    torch.save(sources, f"sources.pt")

    for alpha in args.alpha:
        T = alpha * Tr
        spike_sequences = sample_spike_sequences(args.num_neurons, T, Tr)
        torch.save(spike_sequences, f"spike_sequences_{alpha}.pt")

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
