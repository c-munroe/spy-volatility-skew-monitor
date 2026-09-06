"""
Historical backtest for SPY volatility skew risk reversal strategy.

This script can use either:
- legacy local option daily aggregate CSVs from data/raw/options/YYYY-MM-DD/
- OptionMetrics-style SPY CSV bundles from a sibling SPY_optionmetrics_data folder

Strategy:
- Build daily implied volatility skew from historical option bars
- Measure risk reversal skew = OTM Put IV - OTM Call IV
- Enter when risk reversal skew Z-score is unusually high
- Trade: sell OTM put, buy OTM call, delta hedge with SPY shares
- Exit when skew normalizes or max holding period is reached
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

from skew_strategy import calculate_hedge_shares, get_z_score_trade_action
from implied_vol import implied_vol_newton
from price_data import load_price_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIONMETRICS_DIR = PROJECT_ROOT.parent / "SPY_optionmetrics_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_EQUITY_PLOT = PROJECT_ROOT / "assets" / "backtest_equity_curve.png"
CRASH_FILTER_CHOICES = ("none", "5d-drop", "all")
CRASH_FILTER_LABELS = {
    "none": "Baseline (none)",
    "5d-drop": "5D drop filter",
    "all": "All crash filters",
}


def parse_option_ticker(option_ticker: str) -> dict:
    """
    Parse option ticker like:
    O:SPY251219P00575000

    Returns underlying, expiration, option type, and strike.
    """

    cleaned = option_ticker.replace("_", ":") if option_ticker.startswith("O_") else option_ticker

    pattern = r"O:([A-Z]+)(\d{6})([CP])(\d{8})"
    match = re.match(pattern, cleaned)

    if not match:
        raise ValueError(f"Could not parse option ticker: {option_ticker}")

    underlying = match.group(1)
    expiration_code = match.group(2)
    option_code = match.group(3)
    strike_code = match.group(4)

    yy = expiration_code[:2]
    mm = expiration_code[2:4]
    dd = expiration_code[4:6]

    expiration = f"20{yy}-{mm}-{dd}"
    option_type = "call" if option_code == "C" else "put"
    strike = int(strike_code) / 1000

    return {
        "underlying": underlying,
        "expiration": expiration,
        "option_type": option_type,
        "strike": strike,
    }


def load_option_data(raw_options_dir: str) -> pd.DataFrame:
    """
    Load all option CSVs from one folder into one DataFrame.
    """

    rows = []

    csv_files = sorted(Path(raw_options_dir).glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {raw_options_dir}")

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if df.empty:
            continue

        ticker = df["ticker"].iloc[0]
        metadata = parse_option_ticker(ticker)

        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        df["strike"] = metadata["strike"]
        df["Option_Type"] = metadata["option_type"]
        df["Expiry"] = metadata["expiration"]

        # Prefer VWAP if available, otherwise use close.
        df["Option_Price"] = df["vwap"].fillna(df["close"])

        rows.append(df)

    if not rows:
        raise ValueError("No usable option data loaded.")

    options_df = pd.concat(rows, ignore_index=True)

    options_df = options_df[
        (options_df["Option_Price"].notna())
        & (options_df["Option_Price"] > 0)
    ].copy()

    return options_df


def load_optionmetrics_option_data(
    optionmetrics_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str = "SPY",
    max_dte: int = 75,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Load OptionMetrics option prices into the backtest's normalized schema.

    OptionMetrics stores strike_price in thousandths. The file already includes
    vendor implied volatility, so this loader treats that as Calculated_IV.
    """

    file_path = Path(optionmetrics_dir) / "options_prices.csv.gz"

    if not file_path.exists():
        raise FileNotFoundError(f"OptionMetrics option file not found: {file_path}")

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    usecols = [
        "date",
        "ticker",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "impl_volatility",
        "delta",
        "optionid",
    ]

    rows = []

    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["ticker"] == ticker].copy()

        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"])

        if start_ts is not None:
            chunk = chunk[chunk["date"] >= start_ts]

        if end_ts is not None:
            chunk = chunk[chunk["date"] <= end_ts]

        if chunk.empty:
            continue

        chunk["Expiry"] = pd.to_datetime(chunk["exdate"])
        chunk["DTE"] = (chunk["Expiry"] - chunk["date"]).dt.days
        chunk["Bid"] = pd.to_numeric(chunk["best_bid"], errors="coerce")
        chunk["Ask"] = pd.to_numeric(chunk["best_offer"], errors="coerce")
        chunk["Option_Price"] = (chunk["Bid"] + chunk["Ask"]) / 2
        chunk["Spread"] = chunk["Ask"] - chunk["Bid"]
        chunk["Spread_Pct"] = chunk["Spread"] / chunk["Option_Price"]
        chunk["Calculated_IV"] = pd.to_numeric(
            chunk["impl_volatility"],
            errors="coerce",
        )
        chunk["Vendor_Delta"] = pd.to_numeric(chunk["delta"], errors="coerce")

        chunk = chunk[
            (chunk["DTE"] > 0)
            & (chunk["DTE"] <= max_dte)
            & (chunk["Ask"].notna())
            & (chunk["Bid"].notna())
            & (chunk["Ask"] >= chunk["Bid"])
            & (chunk["Ask"] > 0)
            & (chunk["Option_Price"] > 0)
            & (chunk["Spread_Pct"].notna())
            & (chunk["Calculated_IV"].notna())
            & (chunk["Calculated_IV"] > 0)
            & (chunk["Calculated_IV"] < 5)
        ].copy()

        if chunk.empty:
            continue

        chunk["strike"] = chunk["strike_price"] / 1000
        chunk["Option_Type"] = np.where(chunk["cp_flag"] == "C", "call", "put")
        chunk["T"] = chunk["DTE"] / 365
        chunk = chunk.rename(
            columns={
                "optionid": "Option_ID",
            }
        )

        rows.append(
            chunk[
                [
                    "date",
                    "Expiry",
                    "DTE",
                    "T",
                    "Option_ID",
                    "Option_Type",
                    "strike",
                    "Bid",
                    "Ask",
                    "Option_Price",
                    "Spread_Pct",
                    "Calculated_IV",
                    "Vendor_Delta",
                ]
            ]
        )

    if not rows:
        raise ValueError("No usable OptionMetrics option rows loaded.")

    options_df = pd.concat(rows, ignore_index=True)

    return options_df


def prepare_spy_data(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load SPY historical prices and log returns.
    """

    spy = load_price_data(
        ticker="SPY",
        start=start_date,
        end=end_date,
    )

    spy["Date"] = pd.to_datetime(spy["Date"])

    return spy


def load_optionmetrics_price_data(
    optionmetrics_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Load SPY prices from the OptionMetrics security_prices file.
    """

    file_path = Path(optionmetrics_dir) / "security_prices.csv.gz"

    if not file_path.exists():
        raise FileNotFoundError(f"OptionMetrics security file not found: {file_path}")

    raw = pd.read_csv(file_path, usecols=["date", "ticker", "close", "return"])
    raw = raw[raw["ticker"] == ticker].copy()

    raw["Date"] = pd.to_datetime(raw["date"])

    if start_date:
        raw = raw[raw["Date"] >= pd.to_datetime(start_date)]

    if end_date:
        raw = raw[raw["Date"] <= pd.to_datetime(end_date)]

    raw = raw.sort_values("Date")

    df = raw[["Date", "close", "return"]].rename(
        columns={
            "close": "Close",
            "return": "Return",
        }
    )
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    return df.dropna(subset=["Close"]).reset_index(drop=True)


def attach_spot_prices(
    options_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach daily SPY close prices and moneyness to normalized option rows.
    """

    price_lookup = spy_df.set_index("Date")["Close"]

    options_df = options_df.copy()
    options_df["Spot"] = options_df["date"].map(price_lookup)
    options_df = options_df[options_df["Spot"].notna()].copy()
    options_df["Moneyness"] = options_df["strike"] / options_df["Spot"]

    return options_df


def calculate_daily_iv_surface(
    options_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate implied volatility for every option row.
    """

    results = []

    spy_lookup = spy_df.set_index("Date")

    for _, row in options_df.iterrows():
        current_date = row["date"]

        if current_date not in spy_lookup.index:
            continue

        spot = float(spy_lookup.loc[current_date, "Close"])
        expiry = pd.to_datetime(row["Expiry"])

        days_to_expiry = (expiry - current_date).days

        if days_to_expiry <= 0:
            continue

        T = days_to_expiry / 365

        # Use only log returns available up to the current date.
        log_returns_so_far = spy_df[spy_df["Date"] <= current_date]["Log_Return"]

        try:
            iv = implied_vol_newton(
                market_price=float(row["Option_Price"]),
                S=spot,
                K=float(row["strike"]),
                T=T,
                r=risk_free_rate,
                option_type=row["Option_Type"],
                log_returns=log_returns_so_far,
                tolerance=0.001,
                max_iterations=100,
            )
        except Exception:
            iv = np.nan

        if pd.isna(iv) or iv <= 0 or iv > 5:
            continue

        result = row.to_dict()
        result["Spot"] = spot
        result["T"] = T
        result["Calculated_IV"] = iv
        result["Moneyness"] = float(row["strike"]) / spot

        results.append(result)

    return pd.DataFrame(results)


def closest_row_by_moneyness(
    df: pd.DataFrame,
    option_type: str,
    target_moneyness: float,
) -> pd.Series:
    """
    Find option row closest to target moneyness.
    """

    subset = df[df["Option_Type"] == option_type].copy()

    if subset.empty:
        raise ValueError(f"No {option_type} rows found.")

    idx = (subset["Moneyness"] - target_moneyness).abs().idxmin()

    return subset.loc[idx]


def _option_label(row: pd.Series) -> str:
    """
    Return a readable contract label for logs and output files.
    """

    if "ticker" in row and pd.notna(row["ticker"]):
        return str(row["ticker"])

    if "Option_ID" in row and pd.notna(row["Option_ID"]):
        return f"OM:{int(row['Option_ID'])}"

    return "UNKNOWN"


def _option_id(row: pd.Series) -> int | None:
    """
    Return OptionMetrics option id when available.
    """

    if "Option_ID" not in row or pd.isna(row["Option_ID"]):
        return None

    return int(row["Option_ID"])


def select_daily_risk_reversal_rows(
    group: pd.DataFrame,
    put_moneyness: float,
    call_moneyness: float,
    target_dte: int | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Select same-expiry put/call rows for one day's risk reversal.
    """

    candidates = group.copy()

    if "DTE" in candidates.columns:
        if min_dte is not None:
            candidates = candidates[candidates["DTE"] >= min_dte]

        if max_dte is not None:
            candidates = candidates[candidates["DTE"] <= max_dte]

    if candidates.empty:
        raise ValueError("No option rows passed daily DTE filters.")

    if "Expiry" in candidates.columns:
        expiry_values = pd.Index(candidates["Expiry"].dropna().unique())
        best_pair = None
        best_score = None

        for expiry in expiry_values:
            expiry_group = candidates[candidates["Expiry"] == expiry]

            try:
                put_row = closest_row_by_moneyness(
                    expiry_group,
                    option_type="put",
                    target_moneyness=put_moneyness,
                )
                call_row = closest_row_by_moneyness(
                    expiry_group,
                    option_type="call",
                    target_moneyness=call_moneyness,
                )
            except ValueError:
                continue

            moneyness_score = abs(put_row["Moneyness"] - put_moneyness) + abs(
                call_row["Moneyness"] - call_moneyness
            )
            dte_score = 0.0

            if target_dte is not None and "DTE" in expiry_group.columns:
                dte_score = abs(float(expiry_group["DTE"].median()) - target_dte) / target_dte

            score = moneyness_score + (0.25 * dte_score)

            if best_score is None or score < best_score:
                best_pair = (put_row, call_row)
                best_score = score

        if best_pair is not None:
            return best_pair

        raise ValueError("No same-expiry put/call pair found.")

    put_row = closest_row_by_moneyness(
        candidates,
        option_type="put",
        target_moneyness=put_moneyness,
    )
    call_row = closest_row_by_moneyness(
        candidates,
        option_type="call",
        target_moneyness=call_moneyness,
    )

    return put_row, call_row


def build_daily_skew_history(
    iv_surface_df: pd.DataFrame,
    put_moneyness: float = 0.90,
    call_moneyness: float = 1.10,
    target_dte: int | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> pd.DataFrame:
    """
    Build one daily skew metric per date.

    Risk Reversal Skew = OTM Put IV - OTM Call IV
    """

    rows = []

    for current_date, group in iv_surface_df.groupby("date"):
        try:
            put_row, call_row = select_daily_risk_reversal_rows(
                group,
                put_moneyness=put_moneyness,
                call_moneyness=call_moneyness,
                target_dte=target_dte,
                min_dte=min_dte,
                max_dte=max_dte,
            )

            rr_skew = float(put_row["Calculated_IV"]) - float(call_row["Calculated_IV"])
            option_t = float(put_row["T"]) if "T" in put_row else float(group["T"].iloc[0])

            rows.append(
                {
                    "date": current_date,
                    "Spot": float(group["Spot"].iloc[0]),
                    "Expiry": put_row["Expiry"],
                    "DTE": int(put_row["DTE"]) if "DTE" in put_row else np.nan,
                    "Sell_Put_Option_ID": _option_id(put_row),
                    "Sell_Put_Ticker": _option_label(put_row),
                    "Sell_Put_Strike": float(put_row["strike"]),
                    "Sell_Put_Price": float(put_row["Option_Price"]),
                    "Sell_Put_Mid": float(put_row["Option_Price"]),
                    "Sell_Put_Bid": float(put_row["Bid"]) if "Bid" in put_row else np.nan,
                    "Sell_Put_Ask": float(put_row["Ask"]) if "Ask" in put_row else np.nan,
                    "Sell_Put_Spread_Pct": (
                        float(put_row["Spread_Pct"])
                        if "Spread_Pct" in put_row
                        else np.nan
                    ),
                    "Sell_Put_IV": float(put_row["Calculated_IV"]),
                    "Sell_Put_Vendor_Delta": (
                        float(put_row["Vendor_Delta"])
                        if "Vendor_Delta" in put_row
                        and pd.notna(put_row["Vendor_Delta"])
                        else np.nan
                    ),
                    "Buy_Call_Option_ID": _option_id(call_row),
                    "Buy_Call_Ticker": _option_label(call_row),
                    "Buy_Call_Strike": float(call_row["strike"]),
                    "Buy_Call_Price": float(call_row["Option_Price"]),
                    "Buy_Call_Mid": float(call_row["Option_Price"]),
                    "Buy_Call_Bid": float(call_row["Bid"]) if "Bid" in call_row else np.nan,
                    "Buy_Call_Ask": float(call_row["Ask"]) if "Ask" in call_row else np.nan,
                    "Buy_Call_Spread_Pct": (
                        float(call_row["Spread_Pct"])
                        if "Spread_Pct" in call_row
                        else np.nan
                    ),
                    "Buy_Call_IV": float(call_row["Calculated_IV"]),
                    "Buy_Call_Vendor_Delta": (
                        float(call_row["Vendor_Delta"])
                        if "Vendor_Delta" in call_row
                        and pd.notna(call_row["Vendor_Delta"])
                        else np.nan
                    ),
                    "Risk_Reversal_Skew": rr_skew,
                    "Put_Moneyness": float(put_row["Moneyness"]),
                    "Call_Moneyness": float(call_row["Moneyness"]),
                    "T": option_t,
                }
            )

        except Exception:
            continue

    skew_history = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if skew_history.empty:
        return skew_history

    # Avoid lookahead bias: compare today's skew to prior rolling window.
    rolling_mean = skew_history["Risk_Reversal_Skew"].rolling(30).mean().shift(1)
    rolling_std = skew_history["Risk_Reversal_Skew"].rolling(30).std().shift(1)

    skew_history["Skew_Z"] = (
        (skew_history["Risk_Reversal_Skew"] - rolling_mean) / rolling_std
    )

    return skew_history


def add_crash_filter_features(
    skew_history: pd.DataFrame,
    price_df: pd.DataFrame | None = None,
    drop_window: int = 5,
    drop_threshold: float = -0.03,
    ma_window: int = 200,
    rv_window: int = 20,
    rv_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Add point-in-time crash-regime filters to the daily skew history.

    These filters are used only to block new entries. They do not force exits
    once a trade is already open.
    """

    if skew_history.empty:
        return skew_history

    skew_history = skew_history.sort_values("date").reset_index(drop=True).copy()

    if price_df is not None and not price_df.empty:
        price_features = price_df[["Date", "Close"]].copy()
        price_features = price_features.rename(
            columns={
                "Date": "date",
                "Close": "Spot",
            }
        )
    else:
        price_features = skew_history[["date", "Spot"]].copy()

    price_features = price_features.sort_values("date").reset_index(drop=True)
    price_features["Ret_5D"] = (
        price_features["Spot"] / price_features["Spot"].shift(drop_window) - 1
    )
    price_features["MA_200"] = price_features["Spot"].rolling(
        ma_window,
        min_periods=ma_window,
    ).mean()
    price_features["Log_Return"] = np.log(
        price_features["Spot"] / price_features["Spot"].shift(1)
    )
    price_features["RV_20"] = (
        price_features["Log_Return"].rolling(
            rv_window,
            min_periods=rv_window,
        ).std()
        * np.sqrt(252)
    )
    price_features["RV20_Prior_P90"] = (
        price_features["RV_20"]
        .expanding(min_periods=252)
        .quantile(rv_quantile)
        .shift(1)
    )

    feature_columns = [
        "date",
        "Ret_5D",
        "MA_200",
        "Log_Return",
        "RV_20",
        "RV20_Prior_P90",
    ]
    skew_history = skew_history.merge(
        price_features[feature_columns],
        on="date",
        how="left",
    )

    skew_history["Crash_Filter_5D_Drop"] = skew_history["Ret_5D"] < drop_threshold
    skew_history["Crash_Filter_Below_MA200"] = skew_history["Spot"] < skew_history["MA_200"]
    skew_history["Crash_Filter_High_RV20"] = (
        skew_history["RV_20"] > skew_history["RV20_Prior_P90"]
    )
    skew_history["Crash_Filter_All"] = (
        skew_history["Crash_Filter_5D_Drop"]
        | skew_history["Crash_Filter_Below_MA200"]
        | skew_history["Crash_Filter_High_RV20"]
    )

    return skew_history


def is_crash_filter_active(row: pd.Series, crash_filter: str) -> bool:
    """
    Return whether the configured crash filter blocks a new trade entry.
    """

    if crash_filter == "none":
        return False

    if crash_filter == "5d-drop":
        return bool(row.get("Crash_Filter_5D_Drop", False))

    if crash_filter == "all":
        return bool(row.get("Crash_Filter_All", False))

    raise ValueError(f"Unknown crash filter: {crash_filter}")


def _optional_float(value: object) -> float | None:
    """
    Convert a quote field to float when it is present and finite.
    """

    if value is None or pd.isna(value):
        return None

    value = float(value)

    return value if np.isfinite(value) else None


def _selected_quote_from_row(row: pd.Series, prefix: str) -> dict:
    """
    Build a selected-contract quote dict from the daily skew row.
    """

    mid = row.get(f"{prefix}_Mid", row.get(f"{prefix}_Price"))

    return {
        "Bid": _optional_float(row.get(f"{prefix}_Bid")),
        "Ask": _optional_float(row.get(f"{prefix}_Ask")),
        "Mid": _optional_float(mid),
    }


def get_option_quote(
    options_df: pd.DataFrame,
    ticker: str | None,
    date: pd.Timestamp,
    option_id: int | None = None,
    price_lookup: pd.DataFrame | None = None,
) -> dict | None:
    """
    Get bid, ask, and midpoint for one contract on one exact date.
    """

    if option_id is not None and price_lookup is not None:
        try:
            quote_row = price_lookup.loc[(option_id, date)]
        except KeyError:
            return None

        if isinstance(quote_row, pd.DataFrame):
            quote_row = quote_row.iloc[0]

        quote = {
            "Bid": _optional_float(quote_row.get("Bid")),
            "Ask": _optional_float(quote_row.get("Ask")),
            "Mid": _optional_float(quote_row.get("Option_Price")),
        }
    else:
        if "ticker" not in options_df.columns:
            return None

        row = options_df[
            (options_df["ticker"] == ticker)
            & (options_df["date"] == date)
        ]

        if row.empty:
            return None

        quote_row = row.iloc[0]
        quote = {
            "Bid": _optional_float(quote_row.get("Bid")),
            "Ask": _optional_float(quote_row.get("Ask")),
            "Mid": _optional_float(quote_row.get("Option_Price")),
        }

    return quote if _price_status(quote["Mid"]) else None


def get_execution_price(
    quote: dict | None,
) -> float | None:
    """
    Return the midpoint trade price for one leg.
    """

    if quote is None:
        return None

    return quote["Mid"] if _price_status(quote["Mid"]) else None


def calculate_entry_hedge(
    entry_row: pd.Series,
    contracts: int,
    risk_free_rate: float,
) -> tuple[float, str]:
    """
    Calculate the static hedge using the pre-Phase-2 Black-Scholes method.
    """

    return (
        calculate_hedge_shares(
            spot=float(entry_row["Spot"]),
            put_strike=float(entry_row["Sell_Put_Strike"]),
            call_strike=float(entry_row["Buy_Call_Strike"]),
            T=float(entry_row["T"]),
            risk_free_rate=risk_free_rate,
            put_iv=float(entry_row["Sell_Put_IV"]),
            call_iv=float(entry_row["Buy_Call_IV"]),
            contracts=contracts,
        ),
        "Black-Scholes",
    )


def get_option_price(
    options_df: pd.DataFrame,
    ticker: str | None,
    date: pd.Timestamp,
    option_id: int | None = None,
    price_lookup: pd.Series | pd.DataFrame | None = None,
) -> float | None:
    """
    Get midpoint option price for exact ticker/date.
    """

    if option_id is not None and price_lookup is not None:
        try:
            price = price_lookup.loc[(option_id, date)]
        except KeyError:
            return None

        if isinstance(price, pd.DataFrame):
            price = price["Option_Price"].iloc[0]
        elif isinstance(price, pd.Series) and "Option_Price" in price.index:
            price = price["Option_Price"]
        if isinstance(price, pd.Series):
            price = price.iloc[0]

        return float(price)

    if "ticker" not in options_df.columns:
        return None

    row = options_df[
        (options_df["ticker"] == ticker)
        & (options_df["date"] == date)
    ]

    if row.empty:
        return None

    return float(row["Option_Price"].iloc[0])


def build_exit_price_lookup(
    options_df: pd.DataFrame,
    skew_history: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Build a compact quote lookup for contracts used by the backtest.
    """

    required_columns = {
        "Option_ID",
        "date",
        "Option_Price",
        "Bid",
        "Ask",
    }

    if not required_columns.issubset(options_df.columns):
        return None

    if not {
        "Sell_Put_Option_ID",
        "Buy_Call_Option_ID",
    }.issubset(skew_history.columns):
        return None

    option_ids = pd.concat(
        [
            skew_history["Sell_Put_Option_ID"],
            skew_history["Buy_Call_Option_ID"],
        ],
        ignore_index=True,
    ).dropna()

    if option_ids.empty:
        return None

    option_ids = option_ids.astype("int64").unique()
    price_rows = options_df[options_df["Option_ID"].isin(option_ids)]
    quote_columns = ["Bid", "Ask", "Option_Price"]

    return price_rows.set_index(["Option_ID", "date"])[quote_columns].sort_index()


def filter_exit_price_rows(
    options_df: pd.DataFrame,
    skew_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only prices for contracts selected in the daily risk reversal history.
    """

    required_columns = {
        "Option_ID",
        "date",
        "Option_Price",
    }

    if not required_columns.issubset(options_df.columns):
        return options_df

    if not {
        "Sell_Put_Option_ID",
        "Buy_Call_Option_ID",
    }.issubset(skew_history.columns):
        return options_df

    option_ids = pd.concat(
        [
            skew_history["Sell_Put_Option_ID"],
            skew_history["Buy_Call_Option_ID"],
        ],
        ignore_index=True,
    ).dropna()

    if option_ids.empty:
        return options_df

    option_ids = option_ids.astype("int64").unique()
    keep_columns = [
        column
        for column in ["Option_ID", "date", "Bid", "Ask", "Option_Price"]
        if column in options_df.columns
    ]

    return options_df.loc[
        options_df["Option_ID"].isin(option_ids),
        keep_columns,
    ].copy()


def _price_status(price: float | None) -> bool:
    """
    Return whether an option price is usable under the current midpoint convention.
    """

    return price is not None and np.isfinite(price) and price > 0


def _get_contract_price(
    options_df: pd.DataFrame,
    ticker: str | None,
    date: pd.Timestamp,
    option_id: int | None,
    price_lookup: pd.DataFrame | None,
) -> float | None:
    """
    Get one held contract's exact-date midpoint price.
    """

    quote = get_option_quote(
        options_df=options_df,
        ticker=ticker,
        date=date,
        option_id=option_id,
        price_lookup=price_lookup,
    )

    if quote is None:
        return None

    return quote["Mid"] if _price_status(quote["Mid"]) else None


def refresh_position_quotes(
    open_trade: dict,
    options_df: pd.DataFrame,
    current_date: pd.Timestamp,
    price_lookup: pd.DataFrame | None,
) -> tuple[dict | None, dict | None]:
    """
    Update held-contract last quotes using only prices available on current_date.
    """

    put_quote = get_option_quote(
        options_df=options_df,
        ticker=open_trade["put_ticker"],
        date=current_date,
        option_id=open_trade["put_option_id"],
        price_lookup=price_lookup,
    )
    call_quote = get_option_quote(
        options_df=options_df,
        ticker=open_trade["call_ticker"],
        date=current_date,
        option_id=open_trade["call_option_id"],
        price_lookup=price_lookup,
    )

    if put_quote is not None:
        open_trade["last_put_price"] = put_quote["Mid"]
        open_trade["last_put_quote"] = put_quote
        open_trade["last_put_quote_date"] = current_date

    if call_quote is not None:
        open_trade["last_call_price"] = call_quote["Mid"]
        open_trade["last_call_quote"] = call_quote
        open_trade["last_call_quote_date"] = current_date

    return put_quote, call_quote


def calculate_position_pnl(
    open_trade: dict,
    current_spot: float,
    contracts: int,
    put_price: float | None = None,
    call_price: float | None = None,
) -> dict:
    """
    Calculate PnL for the static-hedged risk reversal.

    Entry prices are realized execution prices. Current prices are midpoint for
    daily MTM and execution-mode prices only when a trade is actually closed.
    """

    if put_price is None:
        put_price = open_trade["last_put_price"]

    if call_price is None:
        call_price = open_trade["last_call_price"]

    put_pnl = (open_trade["entry_put_price"] - put_price) * 100 * contracts
    call_pnl = (call_price - open_trade["entry_call_price"]) * 100 * contracts
    hedge_pnl = open_trade["hedge_shares"] * (
        current_spot - open_trade["entry_spot"]
    )
    total_pnl = put_pnl + call_pnl + hedge_pnl

    return {
        "put_pnl": put_pnl,
        "call_pnl": call_pnl,
        "hedge_pnl": hedge_pnl,
        "total_pnl": total_pnl,
    }


def mark_to_market_position(
    realized_capital: float,
    open_trade: dict | None,
    current_spot: float,
    contracts: int,
) -> dict:
    """
    Return daily marked NAV without realizing open-position PnL.

    Open option legs are valued at midpoint.
    """

    if open_trade is None:
        return {
            "portfolio_value": realized_capital,
            "unrealized_pnl": 0.0,
            "put_unrealized_pnl": 0.0,
            "call_unrealized_pnl": 0.0,
            "hedge_unrealized_pnl": 0.0,
        }

    pnl = calculate_position_pnl(
        open_trade=open_trade,
        current_spot=current_spot,
        contracts=contracts,
    )

    return {
        "portfolio_value": realized_capital + pnl["total_pnl"],
        "unrealized_pnl": pnl["total_pnl"],
        "put_unrealized_pnl": pnl["put_pnl"],
        "call_unrealized_pnl": pnl["call_pnl"],
        "hedge_unrealized_pnl": pnl["hedge_pnl"],
    }


def build_equity_row(
    current_date: pd.Timestamp,
    realized_capital: float,
    open_trade: dict | None,
    current_spot: float,
    contracts: int,
    pending_exit: dict | None,
) -> dict:
    """
    Build one daily accounting row with realized capital and marked NAV.
    """

    mtm = mark_to_market_position(
        realized_capital=realized_capital,
        open_trade=open_trade,
        current_spot=current_spot,
        contracts=contracts,
    )

    row = {
        "Date": current_date,
        "Portfolio_Value": mtm["portfolio_value"],
        "Realized_Capital": realized_capital,
        "Unrealized_PnL": mtm["unrealized_pnl"],
        "Put_Unrealized_PnL": mtm["put_unrealized_pnl"],
        "Call_Unrealized_PnL": mtm["call_unrealized_pnl"],
        "Hedge_Unrealized_PnL": mtm["hedge_unrealized_pnl"],
        "Open_Position": open_trade is not None,
        "Pending_Exit": pending_exit is not None,
        "Pending_Exit_Signal_Date": (
            pending_exit["signal_date"] if pending_exit is not None else pd.NaT
        ),
        "Put_Quote_Date": pd.NaT,
        "Call_Quote_Date": pd.NaT,
        "Accounting_Diff": 0.0,
    }

    if open_trade is not None:
        row["Put_Quote_Date"] = open_trade["last_put_quote_date"]
        row["Call_Quote_Date"] = open_trade["last_call_quote_date"]

    row["Accounting_Diff"] = (
        row["Portfolio_Value"] - row["Realized_Capital"] - row["Unrealized_PnL"]
    )

    return row


def open_trade_from_signal(
    entry_row: pd.Series,
    entry_index: int,
    pending_entry: dict,
    contracts: int,
    risk_free_rate: float,
) -> dict:
    """
    Open a trade on execution day using that day's selected contracts and prices.
    """

    spot = float(entry_row["Spot"])
    hedge_shares, delta_source = calculate_entry_hedge(
        entry_row=entry_row,
        contracts=contracts,
        risk_free_rate=risk_free_rate,
    )
    put_quote = _selected_quote_from_row(entry_row, "Sell_Put")
    call_quote = _selected_quote_from_row(entry_row, "Buy_Call")
    put_price = get_execution_price(
        quote=put_quote,
    )
    call_price = get_execution_price(
        quote=call_quote,
    )

    if put_price is None or call_price is None:
        raise ValueError("Entry midpoint quote is not executable.")

    current_date = entry_row["date"]

    return {
        "execution_mode": "midpoint",
        "delta_source": delta_source,
        "entry_signal_date": pending_entry["signal_date"],
        "entry_signal_index": pending_entry["signal_index"],
        "entry_index": entry_index,
        "entry_date": current_date,
        "entry_spot": spot,
        "entry_z": pending_entry["z"],
        "entry_execution_z": (
            float(entry_row["Skew_Z"]) if pd.notna(entry_row["Skew_Z"]) else np.nan
        ),
        "entry_rr_skew": pending_entry["rr_skew"],
        "entry_execution_rr_skew": float(entry_row["Risk_Reversal_Skew"]),
        "put_ticker": entry_row["Sell_Put_Ticker"],
        "call_ticker": entry_row["Buy_Call_Ticker"],
        "put_option_id": (
            int(entry_row["Sell_Put_Option_ID"])
            if "Sell_Put_Option_ID" in entry_row
            and pd.notna(entry_row["Sell_Put_Option_ID"])
            else None
        ),
        "call_option_id": (
            int(entry_row["Buy_Call_Option_ID"])
            if "Buy_Call_Option_ID" in entry_row
            and pd.notna(entry_row["Buy_Call_Option_ID"])
            else None
        ),
        "entry_put_price": put_price,
        "entry_call_price": call_price,
        "entry_put_bid": put_quote["Bid"],
        "entry_put_ask": put_quote["Ask"],
        "entry_put_mid": put_quote["Mid"],
        "entry_call_bid": call_quote["Bid"],
        "entry_call_ask": call_quote["Ask"],
        "entry_call_mid": call_quote["Mid"],
        "put_vendor_delta": _optional_float(entry_row.get("Sell_Put_Vendor_Delta")),
        "call_vendor_delta": _optional_float(entry_row.get("Buy_Call_Vendor_Delta")),
        "last_put_price": put_quote["Mid"],
        "last_call_price": call_quote["Mid"],
        "last_put_quote": put_quote,
        "last_call_quote": call_quote,
        "last_put_quote_date": current_date,
        "last_call_quote_date": current_date,
        "hedge_shares": hedge_shares,
    }


def close_trade_record(
    open_trade: dict,
    exit_date: pd.Timestamp,
    exit_spot: float,
    exit_put_price: float,
    exit_call_price: float,
    contracts: int,
    exit_reason: str,
    exit_signal_date: pd.Timestamp | None,
    exit_z: float,
    exit_rr_skew: float,
    realized_capital_after_close: float,
    put_quote_date: pd.Timestamp,
    call_quote_date: pd.Timestamp,
    put_quote: dict,
    call_quote: dict,
) -> dict:
    """
    Build a trade-log row from a realized close.
    """

    pnl = calculate_position_pnl(
        open_trade=open_trade,
        current_spot=exit_spot,
        contracts=contracts,
        put_price=exit_put_price,
        call_price=exit_call_price,
    )

    return {
        "Execution_Mode": open_trade["execution_mode"],
        "Delta_Source": open_trade["delta_source"],
        "Entry_Signal_Date": open_trade["entry_signal_date"],
        "Entry_Date": open_trade["entry_date"],
        "Exit_Signal_Date": exit_signal_date,
        "Exit_Date": exit_date,
        "Exit_Reason": exit_reason,
        "Days_Held": max(0, (exit_date - open_trade["entry_date"]).days),
        "Put_Ticker": open_trade["put_ticker"],
        "Call_Ticker": open_trade["call_ticker"],
        "Put_Option_ID": open_trade["put_option_id"],
        "Call_Option_ID": open_trade["call_option_id"],
        "Entry_Spot": open_trade["entry_spot"],
        "Exit_Spot": exit_spot,
        "Entry_Put_Bid": open_trade["entry_put_bid"],
        "Entry_Put_Ask": open_trade["entry_put_ask"],
        "Entry_Put_Mid": open_trade["entry_put_mid"],
        "Entry_Call_Bid": open_trade["entry_call_bid"],
        "Entry_Call_Ask": open_trade["entry_call_ask"],
        "Entry_Call_Mid": open_trade["entry_call_mid"],
        "Entry_Put_Price": open_trade["entry_put_price"],
        "Entry_Call_Price": open_trade["entry_call_price"],
        "Put_Vendor_Delta": open_trade["put_vendor_delta"],
        "Call_Vendor_Delta": open_trade["call_vendor_delta"],
        "Hedge_Shares": open_trade["hedge_shares"],
        "Entry_Z": open_trade["entry_z"],
        "Entry_Execution_Z": open_trade["entry_execution_z"],
        "Exit_Z": exit_z,
        "Entry_RR_Skew": open_trade["entry_rr_skew"],
        "Entry_Execution_RR_Skew": open_trade["entry_execution_rr_skew"],
        "Exit_RR_Skew": exit_rr_skew,
        "Exit_Put_Bid": put_quote["Bid"],
        "Exit_Put_Ask": put_quote["Ask"],
        "Exit_Put_Mid": put_quote["Mid"],
        "Exit_Call_Bid": call_quote["Bid"],
        "Exit_Call_Ask": call_quote["Ask"],
        "Exit_Call_Mid": call_quote["Mid"],
        "Exit_Put_Price": exit_put_price,
        "Exit_Call_Price": exit_call_price,
        "Put_Exit_Quote_Date": put_quote_date,
        "Call_Exit_Quote_Date": call_quote_date,
        "Used_Stale_Exit_Quote": (
            put_quote_date != exit_date or call_quote_date != exit_date
        ),
        "Put_PnL": pnl["put_pnl"],
        "Call_PnL": pnl["call_pnl"],
        "Hedge_PnL": pnl["hedge_pnl"],
        "Total_PnL": pnl["total_pnl"],
        "Portfolio_Value": realized_capital_after_close,
    }


def validate_backtest_accounting(equity_curve: pd.DataFrame) -> None:
    """
    Assert that daily NAV equals realized capital plus unrealized PnL.
    """

    if equity_curve.empty:
        return

    accounting_diff = (
        equity_curve["Portfolio_Value"]
        - equity_curve["Realized_Capital"]
        - equity_curve["Unrealized_PnL"]
    ).abs()

    if (accounting_diff > 1e-8).any():
        raise AssertionError("Portfolio accounting reconciliation failed.")


def run_backtest(
    skew_history: pd.DataFrame,
    options_df: pd.DataFrame,
    initial_capital: float = 100_000,
    contracts: int = 1,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_holding_period_days: int = 10,
    risk_free_rate: float = 0.05,
    crash_filter: str = "5d-drop",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run risk reversal backtest.

    Entry:
    - Signal after close when Skew_Z > entry_z and crash filter is inactive
    - Execute on the next available trading date using that date's prices

    Exit:
    - Signal after close when Skew_Z < exit_z or max holding period is reached
    - Execute on the next available date with both held option quotes
    """

    realized_capital = initial_capital
    open_trade = None
    pending_entry = None
    pending_exit = None
    equity_rows = []
    trade_rows = []
    blocked_entry_opportunities = 0

    skew_history = skew_history.sort_values("date").reset_index(drop=True)
    price_lookup = build_exit_price_lookup(options_df, skew_history)
    final_index = len(skew_history) - 1

    for i, today in skew_history.iterrows():
        current_date = today["date"]
        z = today["Skew_Z"]
        current_spot = float(today["Spot"])
        executed_entry_today = False
        executed_exit_today = False

        # Execute an exit signal from a prior close. If quotes are missing,
        # carry the pending exit until both held contracts have exact-date prices.
        if open_trade is not None and pending_exit is not None:
            put_quote, call_quote = refresh_position_quotes(
                open_trade=open_trade,
                options_df=options_df,
                current_date=current_date,
                price_lookup=price_lookup,
            )
            exit_put_price = get_execution_price(
                quote=put_quote,
            )
            exit_call_price = get_execution_price(
                quote=call_quote,
            )

            if exit_put_price is not None and exit_call_price is not None:
                pnl = calculate_position_pnl(
                    open_trade=open_trade,
                    current_spot=current_spot,
                    contracts=contracts,
                    put_price=exit_put_price,
                    call_price=exit_call_price,
                )
                realized_capital += pnl["total_pnl"]
                trade_rows.append(
                    close_trade_record(
                        open_trade=open_trade,
                        exit_date=current_date,
                        exit_spot=current_spot,
                        exit_put_price=exit_put_price,
                        exit_call_price=exit_call_price,
                        contracts=contracts,
                        exit_reason=pending_exit["reason"],
                        exit_signal_date=pending_exit["signal_date"],
                        exit_z=pending_exit["z"],
                        exit_rr_skew=pending_exit["rr_skew"],
                        realized_capital_after_close=realized_capital,
                        put_quote_date=current_date,
                        call_quote_date=current_date,
                        put_quote=put_quote,
                        call_quote=call_quote,
                    )
                )

                if verbose:
                    print(
                        f"{current_date.date()} CLOSE | "
                        f"Signal: {pending_exit['signal_date'].date()} | "
                        f"Reason: {pending_exit['reason']} | "
                        f"PnL: ${pnl['total_pnl']:,.2f} | "
                        f"Portfolio: ${realized_capital:,.2f}"
                    )

                open_trade = None
                pending_exit = None
                executed_exit_today = True
            elif verbose and not pending_exit.get("wait_reported", False):
                print(
                    f"{current_date.date()} EXIT WAIT | "
                    f"Signal: {pending_exit['signal_date'].date()} | "
                    "missing held-contract quote"
                )
                pending_exit["wait_reported"] = True

        if open_trade is not None:
            refresh_position_quotes(
                open_trade=open_trade,
                options_df=options_df,
                current_date=current_date,
                price_lookup=price_lookup,
            )

        # Execute an entry signal from the prior close using today's selected
        # contracts and today's SPY spot. No final-row signal can execute.
        if open_trade is None and pending_entry is not None:
            open_trade = open_trade_from_signal(
                entry_row=today,
                entry_index=i,
                pending_entry=pending_entry,
                contracts=contracts,
                risk_free_rate=risk_free_rate,
            )

            if verbose:
                print(
                    f"{current_date.date()} OPEN  | "
                    f"Signal: {pending_entry['signal_date'].date()} | "
                    f"Z: {pending_entry['z']:.2f} | "
                    f"Sell {today['Sell_Put_Ticker']} | "
                    f"Buy {today['Buy_Call_Ticker']} | "
                    f"Hedge shares: {open_trade['hedge_shares']:.2f}"
                )

            pending_entry = None
            executed_entry_today = True

        equity_rows.append(
            build_equity_row(
                current_date=current_date,
                realized_capital=realized_capital,
                open_trade=open_trade,
                current_spot=current_spot,
                contracts=contracts,
                pending_exit=pending_exit,
            )
        )

        # Signals are generated after today's close and can only affect a
        # subsequent row. This preserves the shifted rolling Z-score convention.
        if i == final_index:
            continue

        if (
            open_trade is not None
            and pending_exit is None
            and not executed_entry_today
        ):
            rows_held = i - open_trade["entry_index"]
            action = get_z_score_trade_action(
                z_score=z,
                in_position=True,
                entry_z=entry_z,
                exit_z=exit_z,
            )
            should_exit = (
                action["Signal"] == "EXIT_RISK_REVERSAL"
                or rows_held >= max_holding_period_days
            )

            if should_exit:
                exit_reason = (
                    "Skew normalized"
                    if action["Signal"] == "EXIT_RISK_REVERSAL"
                    else "Max holding period reached"
                )
                pending_exit = {
                    "signal_date": current_date,
                    "signal_index": i,
                    "reason": exit_reason,
                    "z": float(z) if pd.notna(z) else np.nan,
                    "rr_skew": float(today["Risk_Reversal_Skew"]),
                }

                if verbose:
                    print(
                        f"{current_date.date()} EXIT SIGNAL | "
                        f"Reason: {exit_reason}"
                    )

        if (
            open_trade is None
            and pending_entry is None
            and not executed_exit_today
        ):
            entry_blocked = is_crash_filter_active(today, crash_filter)
            action = get_z_score_trade_action(
                z_score=z,
                in_position=False,
                entry_z=entry_z,
                exit_z=exit_z,
            )

            if action["Signal"] == "ENTER_RISK_REVERSAL":
                if entry_blocked:
                    blocked_entry_opportunities += 1
                else:
                    pending_entry = {
                        "signal_date": current_date,
                        "signal_index": i,
                        "z": float(z),
                        "rr_skew": float(today["Risk_Reversal_Skew"]),
                    }

                    if verbose:
                        print(
                            f"{current_date.date()} ENTRY SIGNAL | "
                            f"Z: {z:.2f}"
                        )

    if open_trade is not None:
        final_row = skew_history.iloc[-1]
        final_date = final_row["date"]
        final_spot = float(final_row["Spot"])
        put_quote = open_trade["last_put_quote"]
        call_quote = open_trade["last_call_quote"]
        exit_put_price = get_execution_price(
            quote=put_quote,
        )
        exit_call_price = get_execution_price(
            quote=call_quote,
        )

        if exit_put_price is None or exit_call_price is None:
            raise ValueError("End-of-sample position has no executable midpoint quote.")

        pnl = calculate_position_pnl(
            open_trade=open_trade,
            current_spot=final_spot,
            contracts=contracts,
            put_price=exit_put_price,
            call_price=exit_call_price,
        )
        realized_capital += pnl["total_pnl"]
        trade_rows.append(
            close_trade_record(
                open_trade=open_trade,
                exit_date=final_date,
                exit_spot=final_spot,
                exit_put_price=exit_put_price,
                exit_call_price=exit_call_price,
                contracts=contracts,
                exit_reason="End of sample",
                exit_signal_date=(
                    pending_exit["signal_date"] if pending_exit is not None else pd.NaT
                ),
                exit_z=float(final_row["Skew_Z"])
                if pd.notna(final_row["Skew_Z"])
                else np.nan,
                exit_rr_skew=float(final_row["Risk_Reversal_Skew"]),
                realized_capital_after_close=realized_capital,
                put_quote_date=open_trade["last_put_quote_date"],
                call_quote_date=open_trade["last_call_quote_date"],
                put_quote=put_quote,
                call_quote=call_quote,
            )
        )

        if equity_rows:
            equity_rows[-1] = {
                **equity_rows[-1],
                "Portfolio_Value": realized_capital,
                "Realized_Capital": realized_capital,
                "Unrealized_PnL": 0.0,
                "Put_Unrealized_PnL": 0.0,
                "Call_Unrealized_PnL": 0.0,
                "Hedge_Unrealized_PnL": 0.0,
                "Open_Position": False,
                "Pending_Exit": False,
                "Accounting_Diff": 0.0,
            }

        if verbose:
            print(
                f"{final_date.date()} CLOSE | "
                "Reason: End of sample | "
                f"PnL: ${pnl['total_pnl']:,.2f} | "
                f"Portfolio: ${realized_capital:,.2f}"
            )

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    validate_backtest_accounting(equity_curve)

    if verbose and crash_filter != "none":
        print(
            f"Crash filter '{crash_filter}' blocked "
            f"{blocked_entry_opportunities:,} entry opportunities."
        )

    return equity_curve, trades


def calculate_summary_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """
    Calculate standard backtest summary metrics.
    """

    if equity_curve.empty:
        return {
            "Initial_Capital": initial_capital,
            "Final_Value": np.nan,
            "Total_Return": np.nan,
            "Sharpe_Ratio": np.nan,
            "Max_Drawdown": np.nan,
            "Number_of_Trades": 0,
            "Win_Rate": np.nan,
            "Average_PnL": np.nan,
            "Best_Trade": np.nan,
            "Worst_Trade": np.nan,
        }

    final_value = float(equity_curve["Portfolio_Value"].iloc[-1])
    total_return = (final_value / initial_capital) - 1
    daily_returns = (
        equity_curve["Portfolio_Value"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    daily_return_std = daily_returns.std()
    sharpe_ratio = (
        daily_returns.mean() / daily_return_std * np.sqrt(252)
        if daily_return_std and not np.isnan(daily_return_std)
        else np.nan
    )
    running_peak = equity_curve["Portfolio_Value"].cummax()
    drawdown = (equity_curve["Portfolio_Value"] / running_peak) - 1
    max_drawdown = drawdown.min()

    if trades.empty:
        number_of_trades = 0
        win_rate = np.nan
        avg_pnl = np.nan
        best_trade = np.nan
        worst_trade = np.nan
    else:
        number_of_trades = len(trades)
        win_rate = (trades["Total_PnL"] > 0).mean()
        avg_pnl = trades["Total_PnL"].mean()
        best_trade = trades["Total_PnL"].max()
        worst_trade = trades["Total_PnL"].min()

    return {
        "Initial_Capital": initial_capital,
        "Final_Value": final_value,
        "Total_Return": total_return,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "Number_of_Trades": number_of_trades,
        "Win_Rate": win_rate,
        "Average_PnL": avg_pnl,
        "Best_Trade": best_trade,
        "Worst_Trade": worst_trade,
    }


def summarize_results(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> None:
    """
    Print backtest summary.
    """

    if equity_curve.empty:
        print("No equity curve generated.")
        return

    metrics = calculate_summary_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_capital=initial_capital,
    )

    print()
    print("Backtest Summary")
    print("----------------")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value:     ${metrics['Final_Value']:,.2f}")
    print(f"Total Return:    {metrics['Total_Return']:.2%}")
    print(
        f"Sharpe Ratio:    {metrics['Sharpe_Ratio']:.2f}"
        if np.isfinite(metrics["Sharpe_Ratio"])
        else "Sharpe Ratio:    n/a"
    )
    print(
        f"Max Drawdown:    {metrics['Max_Drawdown']:.2%}"
        if np.isfinite(metrics["Max_Drawdown"])
        else "Max Drawdown:    n/a"
    )

    if not trades.empty:
        print(f"Number of Trades: {metrics['Number_of_Trades']}")
        print(f"Win Rate:         {metrics['Win_Rate']:.2%}")
        print(f"Average PnL:      ${metrics['Average_PnL']:,.2f}")
        print(f"Best Trade:       ${metrics['Best_Trade']:,.2f}")
        print(f"Worst Trade:      ${metrics['Worst_Trade']:,.2f}")
    else:
        print("Number of Trades: 0")


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    output_path: str | Path = DEFAULT_EQUITY_PLOT,
    show_plot: bool = False,
) -> Path | None:
    """
    Plot and save portfolio value over time.
    """

    if equity_curve.empty:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve["Date"], equity_curve["Portfolio_Value"])
    plt.title("Historical Risk Reversal Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close()

    return output_path

def plot_crash_filter_comparison(
    equity_curves: dict[str, pd.DataFrame],
    output_path: str | Path = DEFAULT_EQUITY_PLOT,
    show_plot: bool = False,
) -> Path | None:
    """
    Plot baseline and crash-filter equity curves on one chart.
    """

    usable_curves = {
        name: curve for name, curve in equity_curves.items() if not curve.empty
    }

    if not usable_curves:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "none": {"color": "#0F172A", "linestyle": "-"},   # dark navy
        "5d-drop": {"color": "#38BDF8", "linestyle": "-"}, # sky blue
        "all": {"color": "#F97316", "linestyle": "-"},     # orange
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for crash_filter in CRASH_FILTER_CHOICES:
        curve = usable_curves.get(crash_filter)

        if curve is None:
            continue

        ax.plot(
            curve["Date"],
            curve["Portfolio_Value"],
            label=CRASH_FILTER_LABELS.get(crash_filter, crash_filter),
            linewidth=2,
            **styles.get(crash_filter, {}),
        )

    ax.set_title("SPY Risk Reversal Crash Filter Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)

    return output_path


def _format_metric(value: float, formatter: str) -> str:
    """
    Format metric values for console comparison tables.
    """

    if pd.isna(value):
        return "n/a"

    return formatter.format(value)


def print_crash_filter_comparison(comparison: pd.DataFrame) -> None:
    """
    Print a compact comparison table for the crash-filter modes.
    """

    if comparison.empty:
        print("No crash-filter comparison generated.")
        return

    display = comparison[
        [
            "Crash_Filter",
            "Final_Value",
            "Total_Return",
            "Sharpe_Ratio",
            "Max_Drawdown",
            "Number_of_Trades",
            "Win_Rate",
            "Average_PnL",
            "Best_Trade",
            "Worst_Trade",
        ]
    ].copy()
    display["Crash_Filter"] = display["Crash_Filter"].map(CRASH_FILTER_LABELS)
    display = display.rename(
        columns={
            "Crash_Filter": "Strategy",
            "Final_Value": "Final Value",
            "Total_Return": "Total Return",
            "Sharpe_Ratio": "Sharpe",
            "Max_Drawdown": "Max Drawdown",
            "Number_of_Trades": "Trades",
            "Win_Rate": "Win Rate",
            "Average_PnL": "Avg PnL",
            "Best_Trade": "Best Trade",
            "Worst_Trade": "Worst Trade",
        }
    )

    print()
    print("Crash Filter Comparison")
    print("-----------------------")
    print(
        display.to_string(
            index=False,
            formatters={
                "Final Value": lambda value: _format_metric(value, "${:,.2f}"),
                "Total Return": lambda value: _format_metric(value, "{:.2%}"),
                "Sharpe": lambda value: _format_metric(value, "{:.2f}"),
                "Max Drawdown": lambda value: _format_metric(value, "{:.2%}"),
                "Win Rate": lambda value: _format_metric(value, "{:.2%}"),
                "Avg PnL": lambda value: _format_metric(value, "${:,.2f}"),
                "Best Trade": lambda value: _format_metric(value, "${:,.2f}"),
                "Worst Trade": lambda value: _format_metric(value, "${:,.2f}"),
            },
        )
    )


def _safe_filter_name(crash_filter: str) -> str:
    """
    Return a filesystem-safe crash-filter name.
    """

    return crash_filter.replace("-", "_")


def run_crash_filter_comparison(
    skew_history: pd.DataFrame,
    options_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Path | None]:
    """
    Run baseline and crash-protected backtests and plot their equity curves.
    """

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skew_history.to_csv(output_dir / "skew_history.csv", index=False)

    equity_curves = {}
    comparison_rows = []

    for crash_filter in CRASH_FILTER_CHOICES:
        equity_curve, trades = run_backtest(
            skew_history=skew_history,
            options_df=options_df,
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            max_holding_period_days=args.max_holding_period_days,
            risk_free_rate=args.risk_free_rate,
            crash_filter=crash_filter,
            verbose=False,
        )

        safe_name = _safe_filter_name(crash_filter)
        equity_curve.to_csv(output_dir / f"equity_curve_{safe_name}.csv", index=False)
        trades.to_csv(output_dir / f"trades_{safe_name}.csv", index=False)

        metrics = calculate_summary_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=args.initial_capital,
        )
        metrics["Crash_Filter"] = crash_filter
        comparison_rows.append(metrics)
        equity_curves[crash_filter] = equity_curve

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "crash_filter_comparison.csv", index=False)

    plot_path = plot_crash_filter_comparison(
        equity_curves=equity_curves,
        output_path=args.equity_plot,
        show_plot=args.show_plot,
    )

    return comparison, plot_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the historical backtest.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=["optionmetrics", "legacy"],
        default="optionmetrics",
    )
    parser.add_argument(
        "--optionmetrics-dir",
        type=Path,
        default=DEFAULT_OPTIONMETRICS_DIR,
    )
    parser.add_argument(
        "--raw-options-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "options" / "2025-12-19",
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--initial-capital", type=float, default=100_000)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--max-holding-period-days", type=int, default=10)
    parser.add_argument(
        "--crash-filter",
        choices=CRASH_FILTER_CHOICES,
        default="5d-drop",
        help=(
            "Entry guardrail. '5d-drop' blocks new trades after a sharp "
            "5-day SPY selloff. 'all' evaluates all three guards: 5-day "
            "selloff, below the 200D moving average, and 20D realized "
            "volatility above its prior 90th percentile."
        ),
    )
    parser.add_argument(
        "--compare-crash-filters",
        action="store_true",
        help="Run and plot baseline none, 5d-drop, and all crash-filter modes.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.05)
    parser.add_argument("--put-moneyness", type=float, default=0.90)
    parser.add_argument("--call-moneyness", type=float, default=1.10)
    parser.add_argument("--target-dte", type=int, default=45)
    parser.add_argument("--min-dte", type=int, default=21)
    parser.add_argument("--max-dte", type=int, default=75)
    parser.add_argument("--crash-filter-drop-window", type=int, default=5)
    parser.add_argument("--crash-filter-drop-threshold", type=float, default=-0.03)
    parser.add_argument("--crash-filter-ma-window", type=int, default=200)
    parser.add_argument("--crash-filter-rv-window", type=int, default=20)
    parser.add_argument("--crash-filter-rv-quantile", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--equity-plot", type=Path, default=DEFAULT_EQUITY_PLOT)
    parser.add_argument("--show-plot", action="store_true")

    return parser.parse_args()


def build_backtest_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and normalize option rows and daily skew history.
    """

    if args.data_source == "optionmetrics":
        print(f"Loading OptionMetrics data from {args.optionmetrics_dir}...")
        options_df = load_optionmetrics_option_data(
            optionmetrics_dir=args.optionmetrics_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            ticker=args.ticker,
            max_dte=args.max_dte,
        )
        spy_df = load_optionmetrics_price_data(
            optionmetrics_dir=args.optionmetrics_dir,
            start_date=None,
            end_date=args.end_date,
            ticker=args.ticker,
        )
        iv_surface = attach_spot_prices(
            options_df=options_df,
            spy_df=spy_df,
        )
        options_df = iv_surface
    else:
        print(f"Loading legacy option CSVs from {args.raw_options_dir}...")
        options_df = load_option_data(args.raw_options_dir)
        spy_df = prepare_spy_data(
            start_date=args.start_date or "2025-05-31",
            end_date=args.end_date or "2025-12-20",
        )
        iv_surface = calculate_daily_iv_surface(
            options_df=options_df,
            spy_df=spy_df,
            risk_free_rate=args.risk_free_rate,
        )

    print(f"Loaded {len(options_df):,} option rows.")

    skew_history = build_daily_skew_history(
        iv_surface_df=iv_surface,
        put_moneyness=args.put_moneyness,
        call_moneyness=args.call_moneyness,
        target_dte=args.target_dte,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
    )
    skew_history = add_crash_filter_features(
        skew_history=skew_history,
        price_df=spy_df,
        drop_window=args.crash_filter_drop_window,
        drop_threshold=args.crash_filter_drop_threshold,
        ma_window=args.crash_filter_ma_window,
        rv_window=args.crash_filter_rv_window,
        rv_quantile=args.crash_filter_rv_quantile,
    )

    print(f"Built {len(skew_history):,} daily skew observations.")

    if args.data_source == "optionmetrics":
        options_df = filter_exit_price_rows(
            options_df=options_df,
            skew_history=skew_history,
        )
        print(f"Kept {len(options_df):,} selected-contract exit price rows.")

    return options_df, skew_history


def save_backtest_outputs(
    skew_history: pd.DataFrame,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save standard backtest output CSVs.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skew_history.to_csv(output_dir / "skew_history.csv", index=False)
    equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)


def main() -> None:
    """
    Run the configured backtest.
    """

    args = parse_args()
    options_df, skew_history = build_backtest_inputs(args)

    if args.compare_crash_filters:
        comparison, plot_path = run_crash_filter_comparison(
            skew_history=skew_history,
            options_df=options_df,
            args=args,
        )

        print_crash_filter_comparison(comparison)

        print()
        print(f"Saved comparison outputs to {args.output_dir}")
        if plot_path is not None:
            print(f"Saved comparison plot to {plot_path}")

        return

    equity_curve, trades = run_backtest(
        skew_history=skew_history,
        options_df=options_df,
        initial_capital=args.initial_capital,
        contracts=args.contracts,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_holding_period_days=args.max_holding_period_days,
        risk_free_rate=args.risk_free_rate,
        crash_filter=args.crash_filter,
    )

    save_backtest_outputs(
        skew_history=skew_history,
        equity_curve=equity_curve,
        trades=trades,
        output_dir=args.output_dir,
    )

    summarize_results(
        equity_curve=equity_curve,
        trades=trades,
        initial_capital=args.initial_capital,
    )

    plot_path = plot_equity_curve(
        equity_curve=equity_curve,
        output_path=args.equity_plot,
        show_plot=args.show_plot,
    )

    print()
    print(f"Saved outputs to {args.output_dir}")
    if plot_path is not None:
        print(f"Saved equity plot to {plot_path}")


if __name__ == "__main__":
    main()
