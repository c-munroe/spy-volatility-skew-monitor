"""
Data utilities for loading historical price data.

For now, this file pulls daily historical prices from yfinance
and prepares the return columns needed for volatility estimation
and option pricing simulations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def load_price_data(
    ticker: str = "SPY",
    start: str = "2022-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """
    Pull adjusted daily close prices from yfinance.

    Returns a DataFrame with:
    Date
    Close
    Return
    Log_Return
    """

    raw = yf.download(
    ticker,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()

    # The first column after reset_index is the date column,
    # even if pandas calls it "Date" or "index".
    date_col = raw.columns[0]

    df = raw[[date_col, "Close"]].copy()
    df = df.rename(columns={date_col: "Date"})

    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df = df.dropna().reset_index(drop=True)

    return df