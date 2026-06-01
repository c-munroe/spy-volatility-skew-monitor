"""
Implied volatility solver using Newton-Raphson.

This module backs out the volatility that makes the Black-Scholes
theoretical option price match the observed market option price.
"""

from __future__ import annotations

import pandas as pd

from realized_vol import rolling_realized_vol
from black_scholes import black_scholes_call, black_scholes_put
from greeks import vega


def initial_vol_guess(log_returns, window: int = 30) -> float:
    """
    Use recent realized volatility as the initial guess for implied volatility.
    """

    realized_vol = rolling_realized_vol(log_returns, window=window).dropna()

    if realized_vol.empty:
        return 0.20

    sigma0 = realized_vol.iloc[-1]

    if pd.isna(sigma0) or sigma0 <= 0:
        return 0.20

    return float(sigma0)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Route to the correct Black-Scholes pricing function.
    """

    option_type = option_type.lower()

    if option_type == "call":
        return black_scholes_call(S, K, T, r, sigma)

    if option_type == "put":
        return black_scholes_put(S, K, T, r, sigma)

    raise ValueError("option_type must be either 'call' or 'put'")


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    log_returns,
    tolerance: float = 0.001,
    max_iterations: int = 100,
) -> float:
    """
    Estimate implied volatility using Newton-Raphson.
    """

    sigma = initial_vol_guess(log_returns)

    for _ in range(max_iterations):
        theoretical_price = black_scholes_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
        )

        price_diff = theoretical_price - market_price

        if abs(price_diff) < tolerance:
            return sigma

        option_vega = vega(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            per_vol_point=False,
        )

        if option_vega == 0 or pd.isna(option_vega):
            break

        sigma = sigma - price_diff / option_vega

        if sigma <= 0 or pd.isna(sigma) or sigma > 5:
            return float("nan")

    return sigma