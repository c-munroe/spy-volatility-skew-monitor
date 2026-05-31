# SPY Volatility Skew Monitor and Delta-Hedged Risk Reversal Constructor

This project builds a live options analytics system for analyzing SPY volatility skew and constructing a delta-hedged risk reversal trade. The system pulls live options chain data, calculates bid/ask midpoint prices, solves for implied volatility using a Newton-Raphson method, visualizes the volatility skew across strikes, and constructs a trade that sells rich downside volatility, buys upside convexity, and hedges directional exposure with shares of SPY.

## Project Motivation

The Black-Scholes model assumes a constant volatility input across strikes and expirations. In real options markets, implied volatility is not flat. Equity index options often display downside skew, where out-of-the-money (OTM) puts trade at higher implied volatility than comparable OTM calls. This reflects market demand for crash protection and creates a framework for studying relative volatility pricing across the options chain.

This project was built to explore that idea in a systematic way. Instead of only pricing one option, the system computes implied volatility across many strikes and uses that skew structure to construct a volatility-based trade.

## Core Features

* Pulls live SPY options chain data using `yfinance`
* Filters for OTM puts and calls to build a cleaner skew curve
* Uses bid/ask midpoint as the observed market option price
* Prices European calls and puts with Black-Scholes
* Computes Greeks including delta, gamma, and Vega
* Solves for implied volatility using Newton-Raphson iteration
* Uses recent realized volatility as the initial guess for the IV solver
* Builds and plots the calculated implied volatility skew across strikes
* Calculates downside skew and risk reversal skew
* Constructs a delta-hedged risk reversal trade:

  * Sell OTM put
  * Buy OTM call
  * Hedge net option delta using SPY shares

## Strategy Concept

The strategy focuses on identifying when downside skew appears rich. If OTM put implied volatility is meaningfully higher than OTM call implied volatility, the system constructs a risk reversal:

1. Sell an OTM put to collect elevated downside volatility premium.
2. Buy an OTM call to gain upside convexity.
3. Calculate the net delta of the option position.
4. Hedge with SPY shares to reduce directional exposure.

The goal is not to make a simple bullish or bearish stock bet. The goal is to structure a trade around volatility skew while reducing first-order directional exposure through delta hedging.

## File Structure

```text
black_scholes.py      # Black-Scholes call and put pricing functions
greeks.py             # Delta, gamma, and Vega calculations
realized_vol.py       # Rolling realized volatility estimation
price_data.py         # Historical SPY price data and log returns
implied_vol.py        # Newton-Raphson implied volatility solver
vol_surface.py        # Live options chain processing and skew construction
skew_strategy.py      # Skew metrics and delta-hedged risk reversal construction
requirements.txt      # Python package requirements
README.md             # Project documentation
```

## Methodology

### 1. Historical Price Data

The system pulls historical SPY price data and computes daily log returns. These returns are used to estimate rolling realized volatility, which serves as the initial volatility guess for the Newton-Raphson implied volatility solver.

### 2. Implied Volatility Solver

For each option, the market price is estimated using the bid/ask midpoint:

```python
market_price = (bid + ask) / 2
```

The solver then finds the volatility value that makes the Black-Scholes theoretical price match the observed market price:

```text
Black-Scholes Price(sigma) ≈ Market Mid Price
```

The Newton-Raphson update is:

```text
sigma_next = sigma - (theoretical_price - market_price) / Vega
```

### 3. Volatility Skew Construction

The system filters the options chain to use:

```text
OTM puts where strike < spot price
OTM calls where strike > spot price
```

This is done because OTM options are generally more liquid and better reflect current volatility pricing. The system then plots calculated implied volatility against strike price.

### 4. Risk Reversal Construction

The trade constructor selects:

```text
Put strike near 0.90 moneyness
Call strike near 1.10 moneyness
```

It then computes:

```text
Risk Reversal Skew = OTM Put IV - OTM Call IV
```

For the trade:

```text
Sell OTM put
Buy OTM call
Delta hedge with SPY shares
```

The hedge is calculated as:

```text
Hedge Shares = -Net Option Delta × 100 × Number of Contracts
```

A negative hedge share value means the strategy shorts SPY shares. A positive hedge share value means the strategy buys SPY shares.

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
scipy
matplotlib
yfinance
```

## Usage

To build and plot the live SPY volatility skew:

```bash
python vol_surface.py
```

This will:

1. Pull live SPY options data.
2. Select an expiration date.
3. Filter OTM puts and calls.
4. Calculate implied volatility for each contract.
5. Print the resulting skew DataFrame.
6. Plot calculated IV against strike price.

To construct the delta-hedged risk reversal trade, use the functions in `skew_strategy.py` with the skew DataFrame returned by `build_vol_skew()`.

Example:

```python
from vol_surface import build_vol_skew
from skew_strategy import construct_delta_hedged_risk_reversal

skew = build_vol_skew(ticker="SPY")

trade = construct_delta_hedged_risk_reversal(
    skew_df=skew,
    contracts=1,
    put_moneyness=0.90,
    call_moneyness=1.10,
    risk_free_rate=0.05,
)

for key, value in trade.items():
    print(f"{key}: {value}")
```

## Example Output

The trade constructor returns a dictionary containing:

```text
Trade
Expiry
Spot
Contracts
Sell_Put_Strike
Sell_Put_IV
Sell_Put_Mid
Sell_Put_Delta
Buy_Call_Strike
Buy_Call_IV
Buy_Call_Mid
Buy_Call_Delta
Risk_Reversal_Skew
Net_Option_Delta
Net_Share_Delta
Hedge_Shares
Net_Premium_Per_Share
Net_Premium_Total
```

## Limitations

This project focuses on live options-chain analytics and trade construction. It does not include a full historical PnL backtest because free data sources such as `yfinance` do not provide reliable historical options chain data. A true backtest would require historical option quotes for the same contracts from entry to exit, including bid/ask prices, implied volatility, and Greeks.

The system is designed so that historical options data could be added later if available through a paid or institutional data source.

## Future Improvements

Potential extensions include:

* Add daily snapshot logging to build a custom skew history dataset over time
* Compare calculated IV against Yahoo-provided IV
* Add term structure analysis across expirations
* Add smoothing or interpolation across strikes
* Add transaction cost and bid/ask slippage modeling
* Add dynamic delta hedging and rebalance logic
* Build a dashboard for live skew monitoring
* Integrate historical options data for true PnL backtesting

## Resume Summary

Built a live SPY volatility skew monitor and delta-hedged risk reversal trade constructor in Python, using options-chain data, Black-Scholes pricing, Vega-based Newton-Raphson implied volatility solving, OTM option filtering, skew analysis, and Greek-based hedge sizing.
