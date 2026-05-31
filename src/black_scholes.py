"""
Black-Scholes option pricing formulas.

This module prices European call and put options using the
Black-Scholes model.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Price a European call option using Black-Scholes.

    Parameters
    ----------
    S:
        Current underlying price.
    K:
        Strike price.
    T:
        Time to expiration in years.
    r:
        Annual risk-free rate.
    sigma:
        Annualized volatility.

    Returns
    -------
    float
        Call option price.
    """

    # If option is expired, return intrinsic value
    if T <= 0:
        return max(S - K, 0.0)

    # Handles zero/negative volatility
    if sigma <= 0:
        # i.e. if volatility is zero, the stock path is deterministic
        return max(S - K * math.exp(-r * T), 0.0)

    numerator = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
    denominator = sigma * math.sqrt(T)
    d1 = numerator / denominator
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
    return call_price


def black_scholes_put(
    S: float, 
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Price a European put option using Black-Scholes.
    """

    if T <= 0:
        return max(K - S, 0.0)

    if sigma <= 0:
        # Deterministic zero-volatility put value.
        return max(K * math.exp(-r * T) - S, 0.0)

    numerator = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
    denominator = sigma * math.sqrt(T)
    d1 = numerator / denominator
    d2 = d1 - sigma * math.sqrt(T)

    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price