import os
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

# import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.optimize import brentq
# from scipy.special import lambertw
from tqdm import tqdm


class Neuron:
    """A class representing a spiking neuron.

    Attributes:
        idx (int): the neuron index
        num_inputs (int): the number of inputs
        input_beta (float): the input kernel time constant (in ms)
        nominal_threshold (float): the nominal firing threshold
        absolute_refractory (float): the absolute refractory period (in ms)
        relative_refractory (float): the relative refractory period (in ms)
        sources (np.ndarray, optional): the input sources. Defaults to np.full(num_inputs, np.nan).
        delays (np.ndarray, optional): the input delays. Defaults to np.full(num_inputs, np.nan).
        weights (np.ndarray, optional): the input weights. Defaults to np.full(num_inputs, np.nan).
        firing_times (np.ndarray, optional): the firing times. Defaults to an empty np.ndarray.
        status (str): the status of the neuron (None, optimal, infeasible)

    Methods:
        optimize_weights: optimize the input weights for robust memorization of a single prescribed spike train
        optimize_weights_many: optimize the input weights for robust memorization of many prescribed spike trains
        sim: simulate the neuron
    """

    def __init__(
        self,
        idx: int,
        num_inputs: int,
        input_beta: Optional[float] = 1.0,
        nominal_threshold: Optional[float] = 1.0,
        sources: Optional[np.ndarray] = None,
        delays: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        firing_times: Optional[np.ndarray] = None,
    ):
        """_summary_

        Args:
            idx (int): the neuron index
            num_inputs (int): the number of inputs
            input_beta (float): the input kernel time constant [in tau_0]
            nominal_threshold (float): the nominal firing threshold
            sources (np.ndarray, optional): the input sources. Defaults to None.
            delays (np.ndarray, optional): the input delays [in tau_0]. Defaults to None.
            weights (np.ndarray, optional): the input weights. Defaults to None.
            firing_times (np.ndarray, optional): the firing times [in tau_0]. Defaults to None.

        Raises:
            ValueError: if the number of inputs is negative
            ValueError: if the shape of sources is not (num_inputs)
            ValueError: if the shape of delays is not (num_inputs)
            ValueError: if the shape of weights is not (num_inputs)
        """
        self.idx = idx

        if num_inputs < 0:
            raise ValueError("The number of inputs must be non-negative.")
        self.num_inputs = num_inputs

        if input_beta <= 0:
            raise ValueError("The input kernel parameter must be positive.")

        self.input_beta = input_beta
        self.input_kernel = lambda t_: (t_ > 0) * t_ / self.input_beta * np.exp(1 - t_ / self.input_beta)
        self.input_kernel_prime = lambda t_: (t_ > 0) * np.exp(1 - t_ / self.input_beta) * (1 - t_ / self.input_beta) / self.input_beta

        if nominal_threshold <= 0:
            raise ValueError("The nominal firing threshold must be positive.")
        self.nominal_threshold = nominal_threshold

        # if relative_refractory < 0:
        #     raise ValueError("The relative refractory period must be positive.")
        # self.relative_refractory = relative_refractory

        # if absolute_refractory < 0:
        #     raise ValueError("The absolute refractory period must be positive.")
        # 1.0 = absolute_refractory

        # if relative_refractory > 0:
        #     self.refractory_kernel = lambda t_: np.select([t_ > absolute_refractory, t_ > 0], [np.exp(-(t_ - absolute_refractory) / relative_refractory), np.inf], 0)
        #     self.refractory_kernel_prime = lambda t_: np.select(
        #         [t_ > absolute_refractory, t_ > 0],
        #         [-np.exp(-(t_ - absolute_refractory) / relative_refractory) / relative_refractory, np.inf],
        #         0,
        #     )
        # else:
        #     self.refractory_kernel = lambda t_: np.select([t_ > absolute_refractory, t_ > 0], [0, np.inf], 0)
        #     self.refractory_kernel_prime = lambda t_: np.select([t_ > absolute_refractory, t_ > 0], [0, np.inf], 0)

        # self.refractory_kernel = lambda t_: 1e9 * (t_ <= 1) * (t_ > 0)
        # self.refractory_kernel_prime = lambda t_: np.select([t_ > 1.0, t_ > 0], [0, np.inf], 0)

        if sources is None:
            self.sources = np.full(self.num_inputs, np.nan)
        else:
            if sources.shape != (self.num_inputs,):
                raise ValueError("The shape of sources must be (num_inputs).")
            self.sources = sources

        if delays is None:
            self.delays = np.full(self.num_inputs, np.nan)
        else:
            if delays.shape != (self.num_inputs,):
                raise ValueError("The shape of delays must be (num_inputs).")
            self.delays = delays

        if weights is None:
            self.weights = np.full(self.num_inputs, np.nan)
        else:
            if weights.shape != (self.num_inputs,):
                raise ValueError("The shape of weights must be (num_inputs).")
            self.weights = weights

        if firing_times is None:
            self.firing_times = np.array([])
        else:
            self.firing_times = firing_times

        self.status = None

    def optimize_weights(
        self,
        firing_times: np.ndarray,
        input_firing_times: List[np.ndarray],
        period: float,
        firing_area: Optional[float] = 0.2,
        margin_min: Optional[float] = 1.0,
        slope_min: Optional[float] = 2.0, # 0.5 
        weight_bound: Optional[float] = 0.2,
        weight_regularization: Optional[str] = None,
        weight_quantization: Optional[int] = None,
        sampling_res: Optional[float] = 5e-2,
    ):
        """
        Optimize the weights to robustly reproduce multiple prescribed spike trains when fed with specific input spike trains.

        Args:
            firing_times (np.ndarray): the multiple spike trains to reproduce [in tau_0].
            input_firing_times (list of np.ndarrays): the multiple input spike trains [in tau_0].
            period (float): the period of the spike trains [in tau_0].
            firing_area (float, optional): the half-width defining the area around a spike. Defaults to 0.2.
            margin_min (float, optional): the minimum margin to firing threshold in the silent zone. Defaults to 1.0.
            slope_min (float, optional): the minimum slope around a spike [in firing_threshold / tau_0]. Defaults to 2.0.
            weight_bound (float, optional): the (magnitude) weight bound [in firing_threshold]. Defaults to 0.2.
            weight_regularization (str, optional): the weight regularizer. If None, no regularization (except the bounds). Otherwise, should be 'l1' or 'l2'. Defaults to None.
            weight_quantization (int, optional): the weight quantizer. If None, no quantization. Otherwise, should be a positive integer. Defaults to None.
            sampling_res (float, optional): the sampling resolution to set the template constraints [in tau_0]. Defaults to 5e-2.
        """

        if firing_area > 0.5:
            raise ValueError("The firing area must be smaller than tau_0 / 2.")

        model_gp = gp.Model()
        model_gp.setParam("OutputFlag", 0)

        # adaptive_threshold = lambda t_, ft_: self.nominal_threshold + np.nansum(self.refractory_kernel((t_[:, None] - ft_[None, :]) % period), axis=-1)

        # Create (bounded) variables
        if weight_quantization is None:
            # Continuous variables
            weights_gp = model_gp.addMVar(shape=self.num_inputs, lb=-weight_bound, ub=weight_bound)

            input_potential = lambda t_, ift_: np.nansum(self.input_kernel((t_[:, None, None] - ift_[None, :, :]) % period), axis=-1)
            input_potential_prime = lambda t_, ift_: np.nansum(
                self.input_kernel_prime((t_[:, None, None] - ift_[None, :, :]) % period),
                axis=-1,
            )
        else:
            if weight_quantization < 2:
                raise ValueError("The weight quantization must be at least 2.")
            
            # Quantized variables
            dweight = 2 * weight_bound / (weight_quantization - 1)
            weights_gp = model_gp.addMVar(shape=self.num_inputs, lb=-weight_bound / dweight, ub=weight_bound / dweight, vtype=gp.GRB.INTEGER)

            input_potential = lambda t_, ift_: dweight * np.nansum(self.input_kernel((t_[:, None, None] - ift_[None, :, :]) % period), axis=-1)
            input_potential_prime = lambda t_, ift_: dweight * np.nansum(
                self.input_kernel_prime((t_[:, None, None] - ift_[None, :, :]) % period),
                axis=-1,
            )

        # Set regularizer
        if weight_regularization is not None:
            if weight_regularization not in {"l1", "l2"}:
                raise ValueError("The weight regularization must be None, 'l1', or 'l2'.")

            if weight_regularization == "l1":
                lp_norm = model_gp.addVar()
                model_gp.addConstr(lp_norm == gp.norm(weights_gp, 1))

            else:
                lp_norm = model_gp.addVar()
                model_gp.addConstr(lp_norm == gp.norm(weights_gp, 2))

            objective = lp_norm
            model_gp.setObjective(objective, GRB.MINIMIZE)

        # Add constraints
        max_num_spikes = max([fts.size for fts in input_firing_times])
        input_firing_times = np.vstack([np.pad(ft, (max_num_spikes - fts.size, 0), constant_values=np.nan) for fts in input_firing_times])

        sampling_times = np.arange(0, period, sampling_res)
        
        if firing_times.size > 0:
            dist_left = np.min((sampling_times[None, :] - firing_times[:, None]) % period, axis=0)
            dist_right = np.min((firing_times[:, None] - sampling_times[None, :]) % period, axis=0)
            # Lower bounded by the firing threshold at every firing times (most general case)
            model_gp.addConstr(input_potential(firing_times, input_firing_times) @ weights_gp >= self.nominal_threshold)
            # Upper bounded by the firing threshold right before every firing times, i.e., dist_left > 1.0 (no refractory period) and dist_right <= firing_area 
            model_gp.addConstr(input_potential(sampling_times[(dist_left > 1.0) & (dist_right <= firing_area)], input_firing_times) @ weights_gp <= self.nominal_threshold - 1e-6)
            # Upper bounded by the firing threshold minus some margin elsewhere, i.e., dist_left > 1.0 (no refractory period) and dist_right > firing_area
            model_gp.addConstr(input_potential(sampling_times[(dist_left > 1.0) & (dist_right > firing_area)], input_firing_times) @ weights_gp <= self.nominal_threshold - margin_min - 1e-6)
            # Lower bounded by some minimum slope in the vicinity of the firing times, i.e., dist_left <= firing_area or dist_right <= firing_area
            model_gp.addConstr(input_potential_prime(sampling_times[(dist_left <= firing_area) | (dist_right <= firing_area)], input_firing_times) @ weights_gp >= slope_min + 1e-6)
        else:
            model_gp.addConstr(input_potential(sampling_times, input_firing_times) @ weights_gp <= self.nominal_threshold - margin_min - 1e-6)

        # Optimize the weights
        model_gp.optimize()

        if model_gp.status is GRB.OPTIMAL:
            self.status = "optimal"
            self.weights = weights_gp.X
            if weight_quantization is not None:
                self.weights *= dweight
        else:
            self.status = "infeasible"
            # self.weights = weights_gp.X
            self.weights = np.full(self.num_inputs, np.nan)

        model_gp.dispose()

    def optimize_weights_many(
        self,
        firing_times_s: List[np.ndarray],
        input_firing_times_s: List[List[np.ndarray]],
        period: float,
        firing_area: Optional[float] = 0.2,
        margin_min: Optional[float] = 1.0,
        slope_min: Optional[float] = 2.0, # 0.5 
        weight_bound: Optional[float] = 0.2,
        weight_regularization: Optional[str] = None,
        weight_quantization: Optional[int] = None,
        sampling_res: Optional[float] = 5e-2,
    ):
        """Optimize the weights to robustly reproduce multiple prescribed spike trains when fed with specific input spike trains.

        Args:
            firing_times_s (list of np.ndarrays): the multiple spike trains to reproduce [in tau_0].
            input_firing_times_s (list of list of np.ndarrays): the multiple input spike trains [in tau_0].
            period (float): the period of the spike trains [in tau_0].
            firing_area (float, optional): the half-width defining the area around a spike. Defaults to 0.2.
            margin_min (float, optional): the minimum margin to firing threshold in the silent zone. Defaults to 1.0.
            slope_min (float, optional): the minimum slope around a spike [in firing_threshold / tau_0]. Defaults to 2.0.
            weight_bound (float, optional): the (magnitude) weight bound [in firing_threshold]. Defaults to 0.2.
            weight_regularization (str, optional): the weight regularizer. If None, no regularization (except the bounds). Otherwise, should be 'l1' or 'l2'. Defaults to None.
            weight_quantization (int, optional): the weight quantizer. If None, no quantization. Otherwise, should be a positive integer. Defaults to None.
            sampling_res (float, optional): the sampling resolution to set the template constraints [in tau_0]. Defaults to 5e-2.
        """

        if firing_area > 0.5:
            raise ValueError("The firing area must be smaller than tau_0 / 2.")

        model_gp = gp.Model()
        model_gp.setParam("OutputFlag", 0)

        # Create (bounded) variables
        if weight_quantization is None:
            # Continuous variables
            weights_gp = model_gp.addMVar(shape=self.num_inputs, lb=-weight_bound, ub=weight_bound)

            input_potential = lambda t_, ift_: np.nansum(self.input_kernel((t_[:, None, None] - ift_[None, :, :]) % period), axis=-1)
            input_potential_prime = lambda t_, ift_: np.nansum(
                self.input_kernel_prime((t_[:, None, None] - ift_[None, :, :]) % period),
                axis=-1,
            )
        else:
            if weight_quantization < 2:
                raise ValueError("The weight quantization must be at least 2.")
            
            # Quantized variables
            dweight = 2 * weight_bound / (weight_quantization - 1)
            weights_gp = model_gp.addMVar(shape=self.num_inputs, lb=-weight_bound / dweight, ub=weight_bound / dweight, vtype=gp.GRB.INTEGER)

            input_potential = lambda t_, ift_: dweight * np.nansum(self.input_kernel((t_[:, None, None] - ift_[None, :, :]) % period), axis=-1)
            input_potential_prime = lambda t_, ift_: dweight * np.nansum(
                self.input_kernel_prime((t_[:, None, None] - ift_[None, :, :]) % period),
                axis=-1,
            )

        # Set regularizer
        if weight_regularization is not None:
            if weight_regularization not in {"l1", "l2"}:
                raise ValueError("The weight regularization must be None, 'l1', or 'l2'.")

            if weight_regularization == "l1":
                lp_norm = model_gp.addVar()
                model_gp.addConstr(lp_norm == gp.norm(weights_gp, 1))

            else:
                lp_norm = model_gp.addVar()
                model_gp.addConstr(lp_norm == gp.norm(weights_gp, 2))

            objective = lp_norm
            model_gp.setObjective(objective, GRB.MINIMIZE)

        # Add constraints
        sampling_times = np.arange(0, period, sampling_res)

        for firing_times, input_firing_times in zip(firing_times_s, input_firing_times_s):
            max_num_spikes = max([fts.size for fts in input_firing_times])
            input_firing_times = np.vstack([np.pad(fts, (max_num_spikes - fts.size, 0), constant_values=np.nan) for fts in input_firing_times])

            if firing_times.size > 0:
                dist_left = np.min((sampling_times[None, :] - firing_times[:, None]) % period, axis=0)
                dist_right = np.min((firing_times[:, None] - sampling_times[None, :]) % period, axis=0)
                # Lower bounded by the firing threshold at every firing times (most general case)
                model_gp.addConstr(input_potential(firing_times, input_firing_times) @ weights_gp >= self.nominal_threshold)
                # Upper bounded by the firing threshold right before every firing times, i.e., dist_left > 1.0 (no refractory period) and dist_right <= firing_area 
                model_gp.addConstr(input_potential(sampling_times[(dist_left > 1.0) & (dist_right <= firing_area)], input_firing_times) @ weights_gp <= self.nominal_threshold - 1e-6)
                # Upper bounded by the firing threshold minus some margin elsewhere, i.e., dist_left > 1.0 (no refractory period) and dist_right > firing_area
                model_gp.addConstr(input_potential(sampling_times[(dist_left > 1.0) & (dist_right > firing_area)], input_firing_times) @ weights_gp <= self.nominal_threshold - margin_min - 1e-6)
                # Lower bounded by some minimum slope in the vicinity of the firing times, i.e., dist_left <= firing_area or dist_right <= firing_area
                model_gp.addConstr(input_potential_prime(sampling_times[(dist_left <= firing_area) | (dist_right <= firing_area)], input_firing_times) @ weights_gp >= slope_min + 1e-6)
            else:
                model_gp.addConstr(input_potential(sampling_times, input_firing_times) @ weights_gp <= self.nominal_threshold - margin_min - 1e-6)
            
        # Optimize
        model_gp.optimize()

        if model_gp.status is GRB.OPTIMAL:
            self.status = "optimal"
            self.weights = weights_gp.X
            if weight_quantization is not None:
                self.weights *= dweight
        else:
            self.status = "infeasible"
            # self.weights = weights_gp.X
            self.weights = np.full(self.num_inputs, np.nan)

        model_gp.dispose()

    def sim(
        self,
        tmax: float,
        dt: float,
        input_firing_times: List[np.ndarray],
        std_threshold: Optional[float] = 0.0,
    ):
        """Simulate the neuron on a given time range.

        Args:
            tmax (float): time range upper bound [in tau_0].
            dt (float): time step [in tau_0].
            input_firing_times (list of np.ndarray): the (delayed) input firing times [in tau_0].
            std_threshold (float, optional): firing threshold nominal value standard deviation. Defaults to 0.0.

        Raises:
            ValueError: if the time step is larger than the minimum delay of any neuron.
        """
        if dt > np.min(self.delays):
            raise ValueError("The simulation time step must be smaller than the minimum delay.")

        # note: assume no self-loop
        max_num_spikes = max([fts.size for fts in input_firing_times])
        input_firing_times = np.vstack([np.pad(fts, (max_num_spikes - fts.size, 0), constant_values=np.nan) for fts in input_firing_times])

        firing_threshold = np.random.normal(self.nominal_threshold, std_threshold)

        potential = lambda t_: np.inner(self.weights, np.nansum(self.input_kernel((t_ - input_firing_times)), axis=-1))
        adaptive_threshold = lambda t_: firing_threshold + np.nansum(self.refractory_kernel((t_ - self.firing_times)), axis=-1)
        fun = lambda t_: potential(t_) - adaptive_threshold(t_)

        if self.firing_times.size > 0:
            t0 = np.max(self.firing_times) + 1.0
        else:
            t0 = 0.0
            self.firing_times = np.array([-np.inf]) # to avoid checking if the array is empty

        for t in tqdm(np.arange(t0, tmax, dt), desc="Neuron simulation"):
            if t - self.firing_times[-1] <= 1.0:
                continue
            
            if fun(t) > 0:
                # determine the exact firing times
                self.firing_times = np.append(self.firing_times, brentq(fun, t - dt, t))

                # update the firing threshold
                firing_threshold = np.random.normal(self.nominal_threshold, std_threshold)

        self.firing_times = self.firing_times[self.firing_times > -np.inf]