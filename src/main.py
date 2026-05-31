"""
Main entry point for the SPY volatility skew monitor.

This script:
1. Pulls live SPY options data
2. Builds the calculated volatility skew
3. Constructs a delta-hedged risk reversal trade
4. Saves the skew data to CSV
5. Plots the volatility skew
"""

from __future__ import annotations

from vol_surface import build_vol_skew, plot_vol_skew
from skew_strategy import (
    calculate_risk_reversal_skew,
    construct_delta_hedged_risk_reversal,
)


def main() -> None:
    ticker = "SPY"

    print(f"Building live volatility skew for {ticker}...")

    skew = build_vol_skew(
        ticker=ticker,
        min_days_to_expiry=21,
        risk_free_rate=0.05,
        max_spread_pct=0.50,
    )

    if skew.empty:
        raise ValueError("No valid skew data returned.")

    print("\nSkew Data Preview:")
    print(skew.head())
    print()
    print(skew.tail())

    output_file = "spy_vol_skew.csv"
    skew.to_csv(output_file, index=False)
    print(f"\nSaved skew data to {output_file}")

    rr_skew = calculate_risk_reversal_skew(
        skew_df=skew,
        put_moneyness=0.90,
        call_moneyness=1.10,
    )

    print(f"\nRisk Reversal Skew: {rr_skew:.4f}")

    trade = construct_delta_hedged_risk_reversal(
        skew_df=skew,
        contracts=1,
        put_moneyness=0.90,
        call_moneyness=1.10,
        risk_free_rate=0.05,
    )

    print("\nDelta-Hedged Risk Reversal Trade:")
    for key, value in trade.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPlotting volatility skew...")
    plot_vol_skew(skew)


if __name__ == "__main__":
    main()