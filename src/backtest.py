"""
Historical backtest for SPY volatility skew risk reversal strategy.

This script uses downloaded Massive option daily aggregate CSVs from:
data/raw/options/YYYY-MM-DD/

Strategy:
- Build daily implied volatility skew from historical option bars
- Measure risk reversal skew = OTM Put IV - OTM Call IV
- Enter when risk reversal skew Z-score is unusually high
- Trade: sell OTM put, buy OTM call, delta hedge with SPY shares
- Exit when skew normalizes or max holding period is reached
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skew_strategy import calculate_hedge_shares, get_z_score_trade_action
from implied_vol import implied_vol_newton
from price_data import load_price_data


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


def build_daily_skew_history(
    iv_surface_df: pd.DataFrame,
    put_moneyness: float = 0.90,
    call_moneyness: float = 1.10,
) -> pd.DataFrame:
    """
    Build one daily skew metric per date.

    Risk Reversal Skew = OTM Put IV - OTM Call IV
    """

    rows = []

    for current_date, group in iv_surface_df.groupby("date"):
        try:
            put_row = closest_row_by_moneyness(
                group,
                option_type="put",
                target_moneyness=put_moneyness,
            )

            call_row = closest_row_by_moneyness(
                group,
                option_type="call",
                target_moneyness=call_moneyness,
            )

            rr_skew = float(put_row["Calculated_IV"]) - float(call_row["Calculated_IV"])

            rows.append(
                {
                    "date": current_date,
                    "Spot": float(group["Spot"].iloc[0]),
                    "Expiry": group["Expiry"].iloc[0],
                    "Sell_Put_Ticker": put_row["ticker"],
                    "Sell_Put_Strike": float(put_row["strike"]),
                    "Sell_Put_Price": float(put_row["Option_Price"]),
                    "Sell_Put_IV": float(put_row["Calculated_IV"]),
                    "Buy_Call_Ticker": call_row["ticker"],
                    "Buy_Call_Strike": float(call_row["strike"]),
                    "Buy_Call_Price": float(call_row["Option_Price"]),
                    "Buy_Call_IV": float(call_row["Calculated_IV"]),
                    "Risk_Reversal_Skew": rr_skew,
                    "Put_Moneyness": float(put_row["Moneyness"]),
                    "Call_Moneyness": float(call_row["Moneyness"]),
                    "T": float(group["T"].iloc[0]),
                }
            )

        except Exception:
            continue

    skew_history = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Avoid lookahead bias: compare today's skew to prior rolling window.
    rolling_mean = skew_history["Risk_Reversal_Skew"].rolling(30).mean().shift(1)
    rolling_std = skew_history["Risk_Reversal_Skew"].rolling(30).std().shift(1)

    skew_history["Skew_Z"] = (
        (skew_history["Risk_Reversal_Skew"] - rolling_mean) / rolling_std
    )

    return skew_history


def get_option_price(
    options_df: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
) -> float | None:
    """
    Get option price for exact ticker/date.
    """

    row = options_df[
        (options_df["ticker"] == ticker)
        & (options_df["date"] == date)
    ]

    if row.empty:
        return None

    return float(row["Option_Price"].iloc[0])


def run_backtest(
    skew_history: pd.DataFrame,
    options_df: pd.DataFrame,
    initial_capital: float = 100_000,
    contracts: int = 1,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_holding_period_days: int = 10,
    risk_free_rate: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run risk reversal backtest.

    Entry:
    - Enter when Skew_Z > entry_z
    - Sell selected OTM put
    - Buy selected OTM call
    - Delta hedge with SPY shares

    Exit:
    - Exit when Skew_Z < exit_z
    - Or exit when max_holding_period_days is reached
    """

    portfolio_value = initial_capital
    open_trade = None
    equity_rows = []
    trade_rows = []

    skew_history = skew_history.sort_values("date").reset_index(drop=True)

    for i, today in skew_history.iterrows():
        current_date = today["date"]
        z = today["Skew_Z"]

        # -------------------------
        # EXIT LOGIC
        # -------------------------
        if open_trade is not None:
            days_held = i - open_trade["entry_index"]

            action = get_z_score_trade_action(
                z_score=z,
                in_position=True,
                entry_z=entry_z,
                exit_z=exit_z,
            )

            should_exit = (
                action["Signal"] == "EXIT_RISK_REVERSAL"
                or days_held >= max_holding_period_days
            )

            if should_exit:
                exit_put_price = get_option_price(
                    options_df,
                    open_trade["put_ticker"],
                    current_date,
                )

                exit_call_price = get_option_price(
                    options_df,
                    open_trade["call_ticker"],
                    current_date,
                )

                if exit_put_price is None or exit_call_price is None:
                    continue

                exit_spot = float(today["Spot"])

                # Short put PnL: sold at entry, bought back at exit.
                put_pnl = (
                    open_trade["entry_put_price"] - exit_put_price
                ) * 100 * contracts

                # Long call PnL: bought at entry, sold at exit.
                call_pnl = (
                    exit_call_price - open_trade["entry_call_price"]
                ) * 100 * contracts

                # Hedge PnL from SPY share position.
                hedge_pnl = open_trade["hedge_shares"] * (
                    exit_spot - open_trade["entry_spot"]
                )

                total_pnl = put_pnl + call_pnl + hedge_pnl
                portfolio_value += total_pnl

                exit_reason = (
                    "Skew normalized"
                    if action["Signal"] == "EXIT_RISK_REVERSAL"
                    else "Max holding period reached"
                )

                trade_rows.append(
                    {
                        "Entry_Date": open_trade["entry_date"],
                        "Exit_Date": current_date,
                        "Exit_Reason": exit_reason,
                        "Days_Held": days_held,
                        "Put_Ticker": open_trade["put_ticker"],
                        "Call_Ticker": open_trade["call_ticker"],
                        "Entry_Spot": open_trade["entry_spot"],
                        "Exit_Spot": exit_spot,
                        "Entry_Z": open_trade["entry_z"],
                        "Exit_Z": float(z) if pd.notna(z) else np.nan,
                        "Entry_RR_Skew": open_trade["entry_rr_skew"],
                        "Exit_RR_Skew": float(today["Risk_Reversal_Skew"]),
                        "Put_PnL": put_pnl,
                        "Call_PnL": call_pnl,
                        "Hedge_PnL": hedge_pnl,
                        "Total_PnL": total_pnl,
                        "Portfolio_Value": portfolio_value,
                    }
                )

                print(
                    f"{current_date.date()} CLOSE | "
                    f"Reason: {exit_reason} | "
                    f"PnL: ${total_pnl:,.2f} | "
                    f"Portfolio: ${portfolio_value:,.2f}"
                )

                open_trade = None

        # -------------------------
        # ENTRY LOGIC
        # -------------------------
        if open_trade is None:
            action = get_z_score_trade_action(
                z_score=z,
                in_position=False,
                entry_z=entry_z,
                exit_z=exit_z,
            )

            if action["Signal"] == "ENTER_RISK_REVERSAL":
                spot = float(today["Spot"])
                T = float(today["T"])

                hedge_shares = calculate_hedge_shares(
                    spot=spot,
                    put_strike=float(today["Sell_Put_Strike"]),
                    call_strike=float(today["Buy_Call_Strike"]),
                    T=T,
                    risk_free_rate=risk_free_rate,
                    put_iv=float(today["Sell_Put_IV"]),
                    call_iv=float(today["Buy_Call_IV"]),
                    contracts=contracts,
                )

                open_trade = {
                    "entry_index": i,
                    "entry_date": current_date,
                    "entry_spot": spot,
                    "entry_z": float(z),
                    "entry_rr_skew": float(today["Risk_Reversal_Skew"]),
                    "put_ticker": today["Sell_Put_Ticker"],
                    "call_ticker": today["Buy_Call_Ticker"],
                    "entry_put_price": float(today["Sell_Put_Price"]),
                    "entry_call_price": float(today["Buy_Call_Price"]),
                    "hedge_shares": hedge_shares,
                }

                print(
                    f"{current_date.date()} OPEN  | "
                    f"Z: {z:.2f} | "
                    f"Sell {today['Sell_Put_Ticker']} | "
                    f"Buy {today['Buy_Call_Ticker']} | "
                    f"Hedge shares: {hedge_shares:.2f}"
                )

        equity_rows.append(
            {
                "Date": current_date,
                "Portfolio_Value": portfolio_value,
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)

    return equity_curve, trades


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

    final_value = float(equity_curve["Portfolio_Value"].iloc[-1])
    total_return = (final_value / initial_capital) - 1

    print()
    print("Backtest Summary")
    print("----------------")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value:     ${final_value:,.2f}")
    print(f"Total Return:    {total_return:.2%}")

    if not trades.empty:
        win_rate = (trades["Total_PnL"] > 0).mean()
        avg_pnl = trades["Total_PnL"].mean()

        print(f"Number of Trades: {len(trades)}")
        print(f"Win Rate:         {win_rate:.2%}")
        print(f"Average PnL:      ${avg_pnl:,.2f}")
    else:
        print("Number of Trades: 0")


def plot_equity_curve(equity_curve: pd.DataFrame) -> None:
    """
    Plot and save portfolio value over time.
    """

    if equity_curve.empty:
        return

    Path("assets").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve["Date"], equity_curve["Portfolio_Value"])
    plt.title("Historical Risk Reversal Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig("assets/backtest_equity_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    initial_capital = 100_000

    raw_options_dir = "data/raw/options/2025-12-19"

    options_df = load_option_data(raw_options_dir)

    spy_df = prepare_spy_data(
        start_date="2025-05-31",
        end_date="2025-12-20",
    )

    iv_surface = calculate_daily_iv_surface(
        options_df=options_df,
        spy_df=spy_df,
        risk_free_rate=0.05,
    )

    skew_history = build_daily_skew_history(
        iv_surface_df=iv_surface,
        put_moneyness=0.90,
        call_moneyness=1.10,
    )

    equity_curve, trades = run_backtest(
    skew_history=skew_history,
    options_df=options_df,
    initial_capital=initial_capital,
    contracts=1,
    entry_z=2.0,
    exit_z=0.5,
    max_holding_period_days=10,
    risk_free_rate=0.05,
)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    skew_history.to_csv("data/processed/skew_history.csv", index=False)
    equity_curve.to_csv("data/processed/equity_curve.csv", index=False)
    trades.to_csv("data/processed/trades.csv", index=False)

    summarize_results(
        equity_curve=equity_curve,
        trades=trades,
        initial_capital=initial_capital,
    )

    plot_equity_curve(equity_curve)