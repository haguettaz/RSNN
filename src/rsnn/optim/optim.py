from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm.autonotebook import trange

from .gmp import observation_block_forward
from .nuv import box_prior, m_ary_prior
from .utils import bin_error, box_error


def compute_bounded_weights(
        C: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        wlim: Tuple[float, float],
        max_iter: int = 1000, 
        err_tol: float = 1e-3,
        rng: np.random.Generator = None
        ):    

    rng = rng or np.random.default_rng()

    K = C.shape[1]

    wmin = np.full(K, wlim[0])
    wmax = np.full(K, wlim[1])

    mw = rng.uniform(wmin, wmax)
    mz = C @ mw

    for _ in trange(max_iter, desc="Optimization"):
        if np.isnan(mw).any():
            return mw, "nan"

        # compute nuv priors based on posterior means only (no variances)
        mfw, Vfw = box_prior(mw, wmin, wmax, 1)
        mbz, Vbz = box_prior(mz, a, b, 1)

        # compute weights posterior means by forward message passing
        mw = np.copy(mfw)
        Vw = np.diag(Vfw)
        for Cn, mbzn, Vbzn in zip(C, mbz, Vbz):
            mw, Vw = observation_block_forward(Cn, mw, Vw, mbzn, Vbzn)

        # compute potential posterior means
        mz = C @ mw

        # stopping criterion
        if box_error(mw, wmin, wmax) < err_tol and box_error(mz, a, b) < err_tol:
            return mw, "solved"

    return mw, "max_iter"

def compute_bounded_discrete_weights(
        C: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        wlim: Tuple[float, float],
        wlvl: int,
        max_iter: int = 5000, 
        var_tol: float = 1e-3,
        err_tol: float = 1e-3,
        rng: np.random.Generator = None
        ):
    
    rng = rng or np.random.default_rng()
    
    K = C.shape[1]

    wmin = np.full((K, wlvl-1), wlim[0])
    wmax = np.full((K, wlvl-1), wlim[1])

    # weights are parametrized as a mixture of wlvl-1 binary components, i.e., wk = wk,1 + wk,2 + ... + wk,wlvl-1
    mwm = rng.uniform(wmin/(wlvl-1), wmax/(wlvl-1))
    Vwm = (wmax-wmin)/(wlvl-1)**2
    mw, Vw = np.sum(mwm, axis=-1), np.sum(Vwm, axis=-1)
    mz = C @ mw

    for _ in trange(max_iter, desc="Optimization"):
        if np.isnan(mw).any():
            return mw, "nan"

        # compute nuv priors based on posterior means and variances
        mfwm, Vfwm = m_ary_prior(mwm, Vwm, wmin, wmax, wlvl)
        mfw, Vfw = np.sum(mfwm, axis=-1), np.sum(Vfwm, axis=-1)
        mbz, Vbz = box_prior(mz, a, b, 1e-2)

        # compute weights posterior means and variances by forward message passing
        mw = np.copy(mfw)
        Vw = np.diag(Vfw)
        for Cn, mbzn, Vbzn in zip(C, mbz, Vbz):
            mw, Vw = observation_block_forward(Cn, mw, Vw, mbzn, Vbzn)
        
        # weights with prior variance zero are problematic BUT should not be updated anyway
        # selection = np.argwhere(Vfw > 1e-3).flatten()

        # compute the duals for propagation through additive boxes
        Wfw = np.diag(1 / Vfw) # RuntimeWarning: divide by zero encountered in divide
        Wtw = np.diag(Wfw - Wfw @ Vw @ Wfw) # RuntimeWarning: invalid value encountered in matmul
        xitw = Wfw @ (mfw - mw) # RuntimeWarning: invalid value encountered in matmul
        
        # change back to posterior means and variances of weights discrete components
        # weight with prior variance zero are not updated
        mwm = mfwm - Vfwm*xitw[:,None]
        Vwm = Vfwm - Vfwm*Wtw[:,None]*Vfwm
        # print("means", mwm[0], "should be in {", wlim[0]/(wlvl-1), ",", wlim[1]/(wlvl-1), "}")
        # print("vars", Vwm[0])

        mw, Vw = np.sum(mwm, axis=-1), np.sum(Vwm, axis=-1)
        mz = C @ mw

        print("weights", mwm[0], flush=True)
        print("box error", box_error(mz, a, b), flush=True)
        print("bin error", bin_error(mwm, wmin/(wlvl-1), wmax/(wlvl-1)), flush=True)

        if box_error(mz, a, b) < err_tol:
            if bin_error(mwm, wmin/(wlvl-1), wmax/(wlvl-1)) < err_tol:
                print("solved", flush=True)
                return mw, "solved"
        elif Vwm.max() < var_tol:
            print("premature binarization", flush=True)
            return mw, "premature_binarization"

    return mw, "max_iter"