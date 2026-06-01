# SPY Volatility Skew Monitor

A Python-based options analytics project that builds a live SPY volatility skew, solves for implied volatility using Newton-Raphson, constructs delta-hedged risk reversal trades, and includes a prototype historical backtest using contract-level options data.

## Overview

This project analyzes the volatility skew in SPY options. In the Black-Scholes model, volatility is assumed to be constant across strikes, but real options markets often show a skew: out-of-the-money puts usually trade at higher implied volatility than out-of-the-money calls because investors pay a premium for downside protection.

The project pulls live SPY options chain data, calculates implied volatility across strikes, visualizes the volatility skew, and constructs a delta-hedged risk reversal trade by selling rich downside volatility, buying upside convexity, and hedging net option delta with SPY shares.

The project also includes a historical backtest engine that uses contract-level daily options aggregate bars to test Z-score entry and exit rules for the risk reversal strategy.

## Key Features

* Pulls live SPY options chain data using `yfinance`
* Filters for out-of-the-money puts and calls
* Uses bid/ask midpoint as the observed market option price
* Implements Black-Scholes call and put pricing
* Calculates option Greeks including delta, gamma, and Vega
* Solves for implied volatility using Newton-Raphson iteration
* Uses rolling realized volatility as the initial IV guess
* Builds and plots the implied volatility skew across strikes
* Calculates downside skew and risk reversal skew
* Constructs a delta-hedged risk reversal:

  * Sell OTM put
  * Buy OTM call
  * Hedge net delta with SPY shares
* Includes a historical backtest using downloaded contract-level options bars
* Saves backtest outputs including equity curve, trade log, and skew history

## Project Structure

```text
spy-volatility-skew-monitor/
│
├── src/
│   ├── black_scholes.py          # Black-Scholes call and put pricing
│   ├── greeks.py                 # Delta, gamma, and Vega calculations
│   ├── implied_vol.py            # Newton-Raphson implied volatility solver
│   ├── price_data.py             # Historical SPY price data and log returns
│   ├── realized_vol.py           # Rolling realized volatility estimator
│   ├── skew_strategy.py          # Skew metrics, signals, and hedge sizing
│   ├── vol_surface.py            # Live volatility skew builder and plotter
│   ├── download_option_data.py   # Optional contract-level options data downloader
│   ├── backtest.py               # Historical skew strategy backtest
│   └── main.py                   # Main live project entry point
│
├── assets/
│   ├── live_vol_skew_sample.png
│   └── backtest_equity_curve.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Methodology

### 1. Market Price from Bid/Ask Midpoint

For each live option, the market price is estimated using the midpoint between the bid and ask:

```python
market_price = (bid + ask) / 2
```

This midpoint is treated as the observed market option price. The midpoint is used instead of `lastPrice` because options often trade less frequently than stocks, making the last traded price potentially stale.

### 2. Black-Scholes Theoretical Price

The project uses Black-Scholes to calculate the theoretical value of European call and put options.

The implied volatility solver repeatedly compares:

```text
Black-Scholes theoretical price - Market midpoint price
```

The goal is to find the volatility input that makes the theoretical option price match the observed market price.

### 3. Newton-Raphson Implied Volatility Solver

Implied volatility is solved using Newton-Raphson:

```text
sigma_next = sigma - (theoretical_price - market_price) / Vega
```

Vega is used because it measures how much the option price changes with respect to volatility. The solver uses recent rolling realized volatility as the initial guess, which gives the iteration a realistic starting point.

### 4. Volatility Skew Construction

The project filters the options chain to focus on out-of-the-money options:

```text
OTM puts: strike < spot price
OTM calls: strike > spot price
```

It then calculates implied volatility for each contract and plots implied volatility against strike price.

### 5. Delta-Hedged Risk Reversal

The strategy module constructs a risk reversal when downside skew appears rich:

```text
Sell OTM put
Buy OTM call
Delta hedge with SPY shares
```

The hedge is calculated using option deltas:

```text
Hedge Shares = -Net Option Delta × 100 × Number of Contracts
```

A negative hedge share value means the strategy shorts SPY shares. A positive hedge share value means the strategy buys SPY shares.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the full live skew monitor and trade constructor:

```bash
python src/main.py
```

This script will:

1. Pull live SPY options data
2. Select an expiration date
3. Filter for liquid OTM puts and calls
4. Calculate implied volatility across strikes
5. Save the skew data to CSV
6. Calculate risk reversal skew
7. Construct a delta-hedged risk reversal trade
8. Plot the volatility skew

## Example Live Output

The trade constructor returns information such as:

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

## Sample Live Volatility Skew

The live monitor calculates implied volatility across current SPY option strikes and plots the resulting volatility skew. The example below shows the typical equity-index smirk shape, where out-of-the-money puts trade at higher implied volatility than out-of-the-money calls due to demand for downside protection.

![Sample Live SPY Volatility Skew](assets/live_vol_skew_sample.png)

The visible transition around spot reflects the construction method: the live skew uses OTM puts below spot and OTM calls above spot. Because live option-chain data can include bid/ask midpoint noise, stale quotes, dividend effects, and simplified Black-Scholes assumptions, the plot is intended as a practical skew monitor rather than a fully smoothed arbitrage-free volatility surface.

## Historical Backtest

The project includes a prototype historical backtest using contract-level daily options aggregate bars. The backtest constructs a SPY risk reversal by selling an OTM put, buying an OTM call, and delta hedging with SPY shares at trade entry.

The backtest uses:

```text
Risk Reversal Skew = OTM Put IV - OTM Call IV
```

A rolling Z-score is calculated on this skew spread. The strategy enters when skew is unusually steep and exits when skew normalizes or when a maximum holding period is reached.

### Backtest Logic

```text
Entry:
- Skew Z-score > entry threshold
- Sell selected OTM put
- Buy selected OTM call
- Delta hedge with SPY shares

Exit:
- Skew Z-score falls below exit threshold
- Or maximum holding period is reached
```

Raw historical options CSVs are not included in this repository because they are locally downloaded market data. The backtest engine is structured to read local contract-level option files from:

```text
data/raw/options/
```

This folder is excluded from Git tracking.

## Backtest Results

Initial sample results used one SPY expiration and a small strike grid of OTM puts and calls.

```text
Initial Capital: $100,000.00
Final Value:     $100,537.93
Total Return:    0.54%
Number of Trades: 5
Win Rate:         100.00%
Average PnL:      $107.59
```

These results are from a prototype sample and should not be interpreted as production-grade strategy validation. The current backtest does not yet include full bid/ask quote modeling, dynamic delta re-hedging, transaction costs, commissions, or margin requirements for short option positions.

### Backtest Equity Curve

![Backtest Equity Curve](assets/backtest_equity_curve.png)

## Important Note on Trade Execution

This project does not execute live trades through a brokerage API. It constructs and analyzes the trade, including the option legs and required delta hedge. The output is intended for research, learning, and options strategy analysis.

## Limitations

This project includes a prototype historical backtest, but it is not a complete institutional-grade options strategy backtest.

Current limitations include:

* Historical backtest uses daily option aggregate bars rather than full bid/ask quote history
* Raw market data is not included in the repository
* Strategy is tested on a small initial SPY options sample
* Transaction costs, commissions, slippage, and margin requirements are not fully modeled
* Delta hedge is sized at trade entry rather than dynamically rebalanced every day
* Results should be interpreted as a research prototype rather than a proven alpha strategy

## Future Improvements

Potential extensions include:

* Run broader parameter sweeps across entry Z-score, exit Z-score, holding period, and strike moneyness
* Explore ways to increase profitability while preserving high win-rate and risk-controlled behavior
* Add transaction cost, commission, slippage, and bid/ask spread modeling
* Add dynamic delta hedging and daily hedge rebalancing logic
* Compare hedged versus unhedged risk reversal performance
* Estimate max drawdown and worst-trade loss with and without delta hedging
* Expand testing across multiple expirations and larger strike grids
* Add term structure analysis across expirations
* Add interpolation or smoothing across strikes
* Build a dashboard for live skew monitoring and backtest results

## Technologies Used

* Python
* NumPy
* pandas
* SciPy
* matplotlib
* yfinance
* python-dotenv
* massive

## Author

Christopher Munroe
University of Michigan
Mathematics of Finance and Risk Management
[LinkedIn](https://www.linkedin.com/in/chrismunroe12)