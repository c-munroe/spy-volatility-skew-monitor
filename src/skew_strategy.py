"""
Skew strategy utilities.

This module converts calculated implied volatility skews into trading signals.
"""

from __future__ import annotations

import pandas as pd

from greeks import call_delta, put_delta


def get_option_price_from_row(row: pd.Series) -> float:
    """
    Return the best available option price from a row.

    Live data uses Mid.
    Historical aggregate data uses Option_Price.
    """

    if "Mid" in row and pd.notna(row["Mid"]):
        return float(row["Mid"])

    if "Option_Price" in row and pd.notna(row["Option_Price"]):
        return float(row["Option_Price"])

    raise ValueError("Row must contain either Mid or Option_Price.")


def get_atm_iv(skew_df: pd.DataFrame) -> float:
    """
    Find the calculated IV of the option closest to ATM.
    """

    atm_idx = (skew_df["Moneyness"] - 1.0).abs().idxmin()
    return float(skew_df.loc[atm_idx, "Calculated_IV"])


def get_target_otm_put_iv(
    skew_df: pd.DataFrame,
    target_moneyness: float = 0.90,
) -> float:
    """
    Find the calculated IV of the OTM put closest to the target moneyness.
    """

    puts = skew_df[skew_df["Option_Type"] == "put"].copy()

    if puts.empty:
        raise ValueError("No put options found in skew DataFrame.")

    target_idx = (puts["Moneyness"] - target_moneyness).abs().idxmin()

    return float(puts.loc[target_idx, "Calculated_IV"])


def calculate_skew_spread(
    skew_df: pd.DataFrame,
    target_put_moneyness: float = 0.90,
) -> float:
    """
    Calculate downside skew spread.

    Skew Spread = OTM Put IV - ATM IV
    """

    atm_iv = get_atm_iv(skew_df)
    otm_put_iv = get_target_otm_put_iv(
        skew_df,
        target_moneyness=target_put_moneyness,
    )

    return otm_put_iv - atm_iv


def calculate_z_score(
    current_skew_spread: float,
    historical_spreads: pd.Series,
    window: int = 30,
) -> float:
    """
    Calculate rolling Z-score of the current skew spread.
    """

    rolling_mean = historical_spreads.rolling(window=window).mean().iloc[-1]
    rolling_std = historical_spreads.rolling(window=window).std().iloc[-1]

    if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std == 0:
        return 0.0

    return float((current_skew_spread - rolling_mean) / rolling_std)


def generate_skew_signal(
    z_score: float,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    in_position: bool = False,
) -> str:
    """
    Generate a position-aware skew trading signal.

    Strategy logic:
    - If no trade is open, enter when skew is unusually steep.
    - If a trade is open, exit when skew normalizes.

    For this strategy, a high positive Z-score means downside puts are rich
    versus upside calls. The trade is:
    sell OTM put, buy OTM call, and delta hedge.
    """

    if pd.isna(z_score):
        return "HOLD_POSITION" if in_position else "NO_TRADE"

    if in_position:
        if should_exit_trade(z_score, exit_threshold=exit_threshold):
            return "EXIT_RISK_REVERSAL"

        return "HOLD_POSITION"

    if should_enter_trade(z_score, entry_threshold=entry_threshold):
        return "ENTER_RISK_REVERSAL"

    return "NO_TRADE"


def should_enter_trade(
    z_score: float,
    entry_threshold: float = 2.0,
) -> bool:
    """
    Return True when skew is rich enough to open the risk reversal.

    Entry condition:
    Skew_Z > entry_threshold
    """

    return pd.notna(z_score) and z_score > entry_threshold


def should_exit_trade(
    z_score: float,
    exit_threshold: float = 0.5,
) -> bool:
    """
    Return True when skew has normalized enough to close the risk reversal.

    Exit condition:
    Skew_Z < exit_threshold
    """

    return pd.notna(z_score) and z_score < exit_threshold


def get_z_score_trade_action(
    z_score: float,
    in_position: bool,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> dict:
    """
    Return a structured Z-score trading decision.

    This is useful in a backtest or live loop because it includes both the
    action and the reason for that action.
    """

    signal = generate_skew_signal(
        z_score=z_score,
        entry_threshold=entry_z,
        exit_threshold=exit_z,
        in_position=in_position,
    )

    if signal == "ENTER_RISK_REVERSAL":
        reason = f"Skew_Z {z_score:.2f} is above entry threshold {entry_z:.2f}."
    elif signal == "EXIT_RISK_REVERSAL":
        reason = f"Skew_Z {z_score:.2f} is below exit threshold {exit_z:.2f}."
    elif signal == "HOLD_POSITION":
        reason = "Trade is open and skew has not normalized yet."
    else:
        reason = "No open trade and entry threshold has not been reached."

    return {
        "Signal": signal,
        "Z_Score": z_score,
        "In_Position": in_position,
        "Entry_Z": entry_z,
        "Exit_Z": exit_z,
        "Reason": reason,
    }


def get_target_otm_call_row(
    skew_df: pd.DataFrame,
    target_moneyness: float = 1.10,
) -> pd.Series:
    """
    Find the OTM call closest to the target moneyness.

    Example:
    target_moneyness = 1.10 means roughly 10% OTM call.
    """

    calls = skew_df[skew_df["Option_Type"] == "call"].copy()

    if calls.empty:
        raise ValueError("No call options found in skew DataFrame.")

    target_idx = (calls["Moneyness"] - target_moneyness).abs().idxmin()

    return calls.loc[target_idx]


def get_target_otm_put_row(
    skew_df: pd.DataFrame,
    target_moneyness: float = 0.90,
) -> pd.Series:
    """
    Find the OTM put closest to the target moneyness.

    Example:
    target_moneyness = 0.90 means roughly 10% OTM put.
    """

    puts = skew_df[skew_df["Option_Type"] == "put"].copy()

    if puts.empty:
        raise ValueError("No put options found in skew DataFrame.")

    target_idx = (puts["Moneyness"] - target_moneyness).abs().idxmin()

    return puts.loc[target_idx]


def calculate_risk_reversal_skew(
    skew_df: pd.DataFrame,
    put_moneyness: float = 0.90,
    call_moneyness: float = 1.10,
) -> float:
    """
    Calculate risk reversal skew.

    Risk Reversal Skew = OTM Put IV - OTM Call IV

    If this is very positive, puts are much more expensive than calls.
    """

    put_row = get_target_otm_put_row(skew_df, target_moneyness=put_moneyness)
    call_row = get_target_otm_call_row(skew_df, target_moneyness=call_moneyness)

    put_iv = float(put_row["Calculated_IV"])
    call_iv = float(call_row["Calculated_IV"])

    return put_iv - call_iv


def construct_delta_hedged_risk_reversal(
    skew_df: pd.DataFrame,
    contracts: int = 1,
    put_moneyness: float = 0.90,
    call_moneyness: float = 1.10,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Construct the trade:

    Sell OTM put
    Buy OTM call
    Delta hedge with shares of the underlying

    Returns a dictionary describing the trade.
    """

    put_row = get_target_otm_put_row(
        skew_df,
        target_moneyness=put_moneyness,
    )

    call_row = get_target_otm_call_row(
        skew_df,
        target_moneyness=call_moneyness,
    )

    spot = float(skew_df["Spot"].iloc[0])
    T = float(skew_df["T"].iloc[0])
    expiry = skew_df["Expiry"].iloc[0]

    put_strike = float(put_row["strike"])
    call_strike = float(call_row["strike"])

    put_iv = float(put_row["Calculated_IV"])
    call_iv = float(call_row["Calculated_IV"])

    put_mid = get_option_price_from_row(put_row)
    call_mid = get_option_price_from_row(call_row)

    # Long put delta is negative.
    # Since we are SHORT the put, multiply by -1.
    short_put_delta = -put_delta(
        S=spot,
        K=put_strike,
        T=T,
        r=risk_free_rate,
        sigma=put_iv,
    )

    # We are LONG the call, so keep call delta positive.
    long_call_delta = call_delta(
        S=spot,
        K=call_strike,
        T=T,
        r=risk_free_rate,
        sigma=call_iv,
    )

    # Net option delta per share-equivalent.
    net_option_delta = short_put_delta + long_call_delta

    # Each option contract controls 100 shares.
    net_share_delta = net_option_delta * 100 * contracts

    # Hedge shares needed to make total delta approximately zero.
    hedge_shares = -net_share_delta

    # Premium received from selling put and buying call.
    # Positive means you collect premium.
    net_premium_per_share = put_mid - call_mid
    net_premium_total = net_premium_per_share * 100 * contracts

    risk_reversal_skew = put_iv - call_iv

    return {
        "Trade": "Sell OTM Put / Buy OTM Call / Delta Hedge",
        "Expiry": expiry,
        "Spot": spot,
        "Contracts": contracts,

        "Sell_Put_Strike": put_strike,
        "Sell_Put_IV": put_iv,
        "Sell_Put_Mid": put_mid,
        "Sell_Put_Delta": short_put_delta,

        "Buy_Call_Strike": call_strike,
        "Buy_Call_IV": call_iv,
        "Buy_Call_Mid": call_mid,
        "Buy_Call_Delta": long_call_delta,

        "Risk_Reversal_Skew": risk_reversal_skew,
        "Net_Option_Delta": net_option_delta,
        "Net_Share_Delta": net_share_delta,
        "Hedge_Shares": hedge_shares,

        "Net_Premium_Per_Share": net_premium_per_share,
        "Net_Premium_Total": net_premium_total,
    }


def calculate_hedge_shares(
    spot: float,
    put_strike: float,
    call_strike: float,
    T: float,
    risk_free_rate: float,
    put_iv: float,
    call_iv: float,
    contracts: int = 1,
) -> float:
    """
    Calculate SPY shares needed to delta hedge short put + long call.
    """

    short_put_delta = -put_delta(
        S=spot,
        K=put_strike,
        T=T,
        r=risk_free_rate,
        sigma=put_iv,
    )

    long_call_delta = call_delta(
        S=spot,
        K=call_strike,
        T=T,
        r=risk_free_rate,
        sigma=call_iv,
    )

    net_option_delta = short_put_delta + long_call_delta
    net_share_delta = net_option_delta * 100 * contracts

    return -net_share_delta