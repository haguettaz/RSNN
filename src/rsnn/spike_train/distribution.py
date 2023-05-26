
import numpy as np
from scipy.stats import rv_continuous


class LinearDistribution(rv_continuous):
    """
    Linear distribution class.
    """
    def __init__(self, slope:float=0.0, *args, **kwargs):
        """
        Initialize the linear distribution.

        Args:
            slope (float): The slope of the distribution.
        """
        
        super().__init__(*args, **kwargs)
        self.slope = slope

    def _pdf(self, x: np.ndarray):
        """
        Probability density function of the linear distribution.

        Args:
            x (np.ndarray): The value to evaluate the pdf at.

        Returns:
            (np.ndarray): The pdf value at x.
        """
        return self.slope * x 


class SurvivalDistribution(rv_continuous):
    def __init__(self, firing_rate: float, abs_refractory_time: float, rel_refractory_time: float, *args, **kwargs):
        """
        Initialize the spike distribution with the hazard and cumulative hazard of the survival model.

        Args:
            firing_rate (float): The firing rate in [kHz].
            abs_refractory_time (float): The absolute refractory time in [ms].
            rel_refractory_time (float): The relative refractory time in [ms].

        Raises:
            ValueError: If the firing rate is not positive.
            ValueError: If the absolute refractory time is not positive.
            ValueError: If the relative refractory time is not positive.
        """

        super().__init__(*args, **kwargs)

        if firing_rate <= 0:
            raise ValueError("The firing rate must be positive.")
        if abs_refractory_time <= 0:
            raise ValueError("The absolute refractory time must be positive.")
        if rel_refractory_time <= 0:
            raise ValueError("The relative refractory time must be positive.")

        self.hazard = (
            lambda x: (x >= abs_refractory_time)
            * firing_rate
            * (1 - np.exp(-(x - abs_refractory_time) / rel_refractory_time))
        )
        self.Hazard = (
            lambda x: (x >= abs_refractory_time)
            * firing_rate
            * (
                x
                - abs_refractory_time
                + rel_refractory_time * (np.exp(-(x - abs_refractory_time) / rel_refractory_time) - 1)
            )
        )

    def _pdf(self, x: np.ndarray):
        """
        Probability density function of the survival distribution.

        Args:
            x (np.ndarray): The value to evaluate the pdf at.

        Returns:
            (np.ndarray): The pdf value at x.
        """
        return self.hazard(x) * np.exp(-self.Hazard(x))

    def _cdf(self, x: np.ndarray):
        """
        Cumulative density function of the survival distribution.

        Args:
            x (np.ndarray): The value to evaluate the cdf at.

        Returns:
            (np.ndarray): The cdf value at x.
        """
        return 1 - np.exp(-self.Hazard(x))


