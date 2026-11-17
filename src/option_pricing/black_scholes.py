"""Closed-form Black-Scholes prices for European options (ground truth for the project)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def _d1_d2(s, k, tau, r, sigma):
    s, k, tau, r, sigma = (np.asarray(x, dtype=float) for x in (s, k, tau, r, sigma))
    if np.any(s <= 0) or np.any(k <= 0):
        raise ValueError("Spot and strike must be strictly positive.")
    if np.any(tau <= 0) or np.any(sigma <= 0):
        raise ValueError("Maturity and volatility must be strictly positive.")
    vol_sqrt_t = sigma * np.sqrt(tau)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t, s, k, tau, r


def bs_call_price(
    s: ArrayLike, k: ArrayLike, tau: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> np.ndarray:
    """Black-Scholes price of a European call.

    Args:
        s: Spot price of the underlying.
        k: Strike price.
        tau: Time to maturity in years.
        r: Continuously compounded risk-free rate.
        sigma: Annualised volatility.

    Returns:
        Call price, broadcast over the inputs.

    Raises:
        ValueError: If spot, strike, maturity or volatility is not strictly positive.
    """
    d1, d2, s, k, tau, r = _d1_d2(s, k, tau, r, sigma)
    return s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2)


def bs_put_price(
    s: ArrayLike, k: ArrayLike, tau: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> np.ndarray:
    """Black-Scholes price of a European put (same arguments as `bs_call_price`)."""
    d1, d2, s, k, tau, r = _d1_d2(s, k, tau, r, sigma)
    return k * np.exp(-r * tau) * norm.cdf(-d2) - s * norm.cdf(-d1)


def scaled_call_price(m: ArrayLike, tau: ArrayLike, r: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Scaled call price ``pi = C / K`` as a function of moneyness ``m = S0 / K``.

    This is the regression target of the project: with ``K = 1`` and ``S0 = m``.
    """
    return bs_call_price(m, 1.0, tau, r, sigma)
