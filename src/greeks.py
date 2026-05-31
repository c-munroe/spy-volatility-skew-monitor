"""
Option Greeks for European options under Black-Scholes.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")

    if T <= 0:
        raise ValueError("T must be positive for Greeks")

    if sigma <= 0:
        raise ValueError("sigma must be positive for Greeks")

    numerator = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
    denominator = sigma * math.sqrt(T)
    d1 = numerator / denominator

    return d1


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call delta.
    """

    d1 = _d1(S, K, T, r, sigma)

    delta = norm.cdf(d1)

    return delta


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes put delta
    """

    d1 = _d1(S, K, T, r, sigma)

    delta = norm.cdf(d1) - 1

    return delta


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes gamma; same for puts and calls
    """

    d1 = _d1(S, K, T, r, sigma)

    gamma_val = norm.pdf(d1) / (S * sigma * math.sqrt(T))

    return gamma_val


def vega(S: float, K: float, T: float, r: float, sigma: float, per_vol_point: bool = True) -> float:
    """
    Calculate Black-Scholes vega
    If per_vol_point=True, return vega per 1 percentage point change in volatility.
    i.e. sigma moves from 0.20 to 0.21.
    """

    d1 = _d1(S, K, T, r, sigma)

    vega_val = S * norm.pdf(d1) * math.sqrt(T)

    if per_vol_point:
        vega_val = vega_val / 100

    return vega_val