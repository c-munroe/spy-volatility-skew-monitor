"""
Volatility estimation utilities.

This module estimates realized volatility from historical log returns.
For now, we use simple rolling realized volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def rolling_realized_vol(
    log_returns: pd.Series,
    window: int = 30,
) -> pd.Series:
    """
    Estimate annualized realized volatility from rolling daily log returns.

    Parameters
    ----------
    log_returns:
        A pandas Series of daily log returns.

    window:
        Number of trading days used in the rolling volatility estimate.

    Returns
    -------
    pd.Series
        Annualized rolling realized volatility.
    """

    if window <= 1:
        raise ValueError("window must be greater than 1")

    daily_rolling_std = log_returns.rolling(window=window).std()

    annualized_vol = daily_rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    return annualized_vol