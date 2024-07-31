# Recurrent Spiking Neural Networks

*"Et tout d'un coup le souvenir m'est apparu."* - Marcel Proust, *Du côté de chez Swann*

### :boom: Spiking Neural Network

We consider a network of $L$ spiking neurons, each neuron receives $K$ input signals from the other neurons.
The $\ell$-th neuron has potential

$$
    z_{\ell}(t) = \sum_{k=1}^K w_{\ell,k} y_{\ell, k} (t),
$$

with $y_{\ell, k} (t) \triangleq (h_{\ell, k} * x_{i_{\ell, k}})(t)$, and, a spike is produced whenever this potential is above some firing threshold $\theta(t) > 0$.

### :clock9: Precise and Robust Timing

### :link: Associative Recall

## Installation

## Notebooks

- spike trains
- neural network
- optimization
- simulation

## Scripts

## References

[1] H. Aguettaz and H.-A. Loeliger, "Continuous-Time Neural Networks Can Stably Memorize Random Spike Trains", arXiv, 2004 

