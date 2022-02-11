import numpy as np
from scipy.special import lambertw

from rsnn.rsnn.utils import is_memorized


class RSNN:
    def __init__(self, num_neurons, num_synapses, w_lim=(-1.0, 1.0), Th=5, Tr=2, theta=0.0):
        self.theta = 0.0
        self.spike_generator = lambda z_: z_ >= theta
        self.Tr = Tr
        self.num_neurons = num_neurons

        self.init_synapses(num_neurons, num_synapses, w_lim, Tr)
        self.init_impulse_response(Th)
        self.init_learning(w_lim)

    def init_synapses(self, num_neurons, num_synapses, w_lim, Tr):
        self.origins = np.random.randint(0, num_neurons, (num_neurons, num_synapses))
        self.delays = np.random.uniform(0, Tr, (num_neurons, num_synapses))
        self.weights = np.random.uniform(*w_lim, (num_neurons, num_synapses))

    def init_impulse_response(self, Th, eps=1e-3):
        beta = -Th / np.real_if_close(lambertw(-eps / np.exp(1), -1))
        self.impulse_response = lambda t: t / beta * np.exp(1 - t / beta)
        self.grid = np.arange(-Th, 0)
        self.Th = Th

    def init_learning(self, w_lim, alpha=10):
        w_min, w_max = w_lim
        self.grad_loss_fit = (
            lambda w_, z_, y_: 1
            / y_.size
            * np.dot(np.sign(np.dot(z_, w_) - self.theta) + (-1) ** y_, z_)
        )
        self.grad_loss_reg = lambda w_: np.sign(w_ - w_min) + np.sign(w_ - w_max)
        self.grad_loss = lambda w_, z_, y_: self.grad_loss_fit(
            w_, z_, y_
        ) + alpha * self.grad_loss_reg(w_)

    def memorize(self, data, eta=1e-2, num_iter=50000):
        length = data.shape[1]
        z = np.sum(
            [
                self.impulse_response(-self.grid[None, None, :] - self.delays[:, :, None])
                * np.roll(data[self.origins], -l, axis=-1)[:, :, -self.grid.shape[0] :]
                for l in range(length)
            ],
            axis=-1,
        )

        for n in range(self.num_neurons):
            if self.Tr > 0:
                free_indices = [
                    l for l in range(length) if np.sum(np.roll(data[n], -l)[..., -self.Tr :]) < 1
                ]
            else:
                free_indices = np.arange(length)

            for _ in range(num_iter):
                grad_loss = self.grad_loss(
                    self.weights[n], z[free_indices, n], data[n, free_indices]
                )
                if np.abs(grad_loss).max() < 1e-5:
                    print(f"Weights optimization on neuron {n} is done!")
                    break
                self.weights[n] -= eta * grad_loss

        if not is_memorized(self, data):
            print("Error in memorization...")
            return

        print("Memorization is done!")

    def forward(self, input):
        if input.shape[1] < self.Th:
            raise ValueError("Input duration is not sufficiently large")

        input = input[:, -self.Th :]

        output = np.zeros(self.num_neurons, int)
        for n in range(self.num_neurons):
            if self.Tr > 0 and np.sum(input[n, -self.Tr :]) > 0:
                continue

            zn = np.sum(
                self.impulse_response(-self.grid[None, :] - self.delays[n, :, None])
                * input[self.origins[n]],
                axis=-1,
            )
            output[n] = self.spike_generator(np.inner(zn, self.weights[n]))

        return output

