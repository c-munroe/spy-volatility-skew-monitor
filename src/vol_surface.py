"""
Volatility skew / surface builder.

This module pulls a live options chain, filters for liquid OTM options,
calculates implied volatility using Newton-Raphson, and plots the skew.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from price_data import load_price_data
from implied_vol import implied_vol_newton


def year_fraction_to_expiry(expiry: str, valuation_date: date | None = None) -> float:
    """
    Convert an expiration date into time to expiry in years.
    """

    if valuation_date is None:
        valuation_date = date.today()

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - valuation_date).days

    if days_to_expiry <= 0:
        raise ValueError("Expiration must be after today's date.")

    return days_to_expiry / 365


def select_expiry(
    ticker_obj: yf.Ticker,
    expiry: str | None = None,
    min_days_to_expiry: int = 21,
) -> str:
    """
    Select an expiration date.

    If expiry is provided, use it.
    Otherwise, pick the first expiration at least min_days_to_expiry days away.
    """

    expiries = list(ticker_obj.options)

    if not expiries:
        raise ValueError("No option expirations found.")

    if expiry is not None:
        if expiry not in expiries:
            raise ValueError(f"{expiry} is not an available expiration.")
        return expiry

    today = date.today()

    for exp in expiries:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        days = (exp_date - today).days

        if days >= min_days_to_expiry:
            return exp

    return expiries[-1]


def get_spot_price(ticker_obj: yf.Ticker) -> float:
    """
    Get the latest underlying price.
    """

    hist = ticker_obj.history(period="5d", auto_adjust=True)

    if hist.empty:
        raise ValueError("Could not fetch spot price.")

    return float(hist["Close"].dropna().iloc[-1])


def prepare_otm_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot_price: float,
    max_spread_pct: float = 0.50,
) -> pd.DataFrame:
    """
    Filter the option chain to OTM options and calculate bid/ask mid-price.

    For skew construction:
    - use OTM puts where strike < spot
    - use OTM calls where strike > spot
    """

    calls = calls.copy()
    puts = puts.copy()

    calls["Option_Type"] = "call"
    puts["Option_Type"] = "put"

    otm_calls = calls[calls["strike"] > spot_price]
    otm_puts = puts[puts["strike"] < spot_price]

    options = pd.concat([otm_puts, otm_calls], ignore_index=True)

    options["Mid"] = (options["bid"] + options["ask"]) / 2
    options["Spread"] = options["ask"] - options["bid"]
    options["Spread_Pct"] = options["Spread"] / options["Mid"]

    options = options[
        (options["bid"] > 0)
        & (options["ask"] > 0)
        & (options["ask"] >= options["bid"])
        & (options["Mid"] > 0)
        & (options["Spread_Pct"] <= max_spread_pct)
    ].copy()

    if "impliedVolatility" in options.columns:
        options = options.rename(columns={"impliedVolatility": "Yahoo_IV"})

    return options


def build_vol_skew(
    ticker: str = "SPY",
    expiry: str | None = None,
    min_days_to_expiry: int = 21,
    risk_free_rate: float = 0.05,
    vol_window: int = 30,
    max_spread_pct: float = 0.50,
) -> pd.DataFrame:
    """
    Build a calculated implied volatility skew for one expiration.

    Returns a DataFrame with strike, option type, bid, ask, mid, and calculated IV.
    """

    ticker_obj = yf.Ticker(ticker)

    spot_price = get_spot_price(ticker_obj)

    selected_expiry = select_expiry(
        ticker_obj=ticker_obj,
        expiry=expiry,
        min_days_to_expiry=min_days_to_expiry,
    )

    T = year_fraction_to_expiry(selected_expiry)

    chain = ticker_obj.option_chain(selected_expiry)

    options = prepare_otm_options(
        calls=chain.calls,
        puts=chain.puts,
        spot_price=spot_price,
        max_spread_pct=max_spread_pct,
    )

    today = date.today()
    history_start = (today - timedelta(days=365)).isoformat()
    history_end = (today + timedelta(days=1)).isoformat()

    price_df = load_price_data(
        ticker=ticker,
        start=history_start,
        end=history_end,
    )

    log_returns = price_df["Log_Return"]

    calculated_ivs = []

    for _, row in options.iterrows():
        try:
            iv = implied_vol_newton(
                market_price=float(row["Mid"]),
                S=spot_price,
                K=float(row["strike"]),
                T=T,
                r=risk_free_rate,
                option_type=row["Option_Type"],
                log_returns=log_returns,
                tolerance=0.001,
                max_iterations=100,
            )
        except Exception:
            iv = np.nan

        calculated_ivs.append(iv)

    options["Calculated_IV"] = calculated_ivs
    options["Moneyness"] = options["strike"] / spot_price
    options["Spot"] = spot_price
    options["Expiry"] = selected_expiry
    options["T"] = T

    options = options[
        (options["Calculated_IV"].notna())
        & (options["Calculated_IV"] > 0.03)
        & (options["Calculated_IV"] < 2.00)
    ].copy()

    columns_to_keep = [
        "contractSymbol",
        "Option_Type",
        "strike",
        "bid",
        "ask",
        "Mid",
        "Spread_Pct",
        "Calculated_IV",
        "Moneyness",
        "Spot",
        "Expiry",
        "T",
    ]

    if "Yahoo_IV" in options.columns:
        columns_to_keep.append("Yahoo_IV")

    available_columns = [col for col in columns_to_keep if col in options.columns]

    return options[available_columns].sort_values("strike").reset_index(drop=True)


def plot_vol_skew(skew_df: pd.DataFrame) -> None:
    """
    Plot calculated implied volatility against strike price.
    """

    if skew_df.empty:
        raise ValueError("No skew data to plot.")

    spot_price = skew_df["Spot"].iloc[0]
    expiry = skew_df["Expiry"].iloc[0]

    plt.figure(figsize=(10, 6))

    for option_type, group in skew_df.groupby("Option_Type"):
        plt.plot(
            group["strike"],
            group["Calculated_IV"],
            marker="o",
            linestyle="",
            label=option_type,
        )

    plt.axvline(
        spot_price,
        linestyle="--",
        label=f"Spot = {spot_price:.2f}",
    )

    plt.title(f"Calculated Volatility Skew for {expiry}")
    plt.xlabel("Strike Price")
    plt.ylabel("Calculated Implied Volatility")
    plt.legend()
    plt.grid(True)

    Path("assets").mkdir(parents=True, exist_ok=True)
    plt.savefig("assets/live_vol_skew_sample.png", dpi=300, bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    skew = build_vol_skew(ticker="SPY")

    print(skew.head())
    print()
    print(skew.tail())

    plot_vol_skew(skew)