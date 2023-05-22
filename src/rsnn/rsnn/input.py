from typing import Callable

import numpy as np

from .neuron import Neuron


class Input:
    """
    Input class for RSNN.
    Can be used for both synaptic input and refractory feedback loop.
    """
    def __init__(self, source:Neuron, delay:float, weight:float, resp:Callable, resp_deriv:Callable):
        """
        Args:
            source (Neuron): the source neuron.
            delay (float): the transmission delay in [ms].
            weight (float): the input weight in [theta].
            resp (Callable): the impulse response.
            resp_deriv (Callable): the derivative of the impulse response.
        """
        self.source = source
        self.weight = weight
        self.delay = delay
        self.resp = resp
        self.resp_deriv = resp_deriv
            
    def signal(self, t:float):
        """
        Returns the instantaneous signal at time t.
        
        Args:
            t (float): the time in [ms].

        Returns:
            (float): the signal value.
        """
        return self.weight * np.sum(self.resp(t - self.delay - self.source.firing_times))
                                    
    def signal_deriv(self, t:float):
        """
        Returns the instantaneous signal derivative at time t.

        Args:
            t (float): the time in [ms].

        Returns:
            (float): the signal derivative value.
        """
        return self.weight * np.sum(self.resp_deriv(t - self.delay - self.source.firing_times)) 