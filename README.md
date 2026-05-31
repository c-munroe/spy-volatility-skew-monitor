# SPY Volatility Skew Monitor

A Python-based options analytics project that builds a live SPY volatility skew, solves for implied volatility using Newton-Raphson, and constructs a delta-hedged risk reversal trade designed to isolate volatility skew exposure.

## Overview

This project analyzes the volatility skew in SPY options. In the Black-Scholes model, volatility is assumed to be constant across strikes, but real options markets often show a skew: out-of-the-money puts usually trade at higher implied volatility than out-of-the-money calls because investors pay a premium for downside protection.

The project pulls live SPY options chain data, calculates implied volatility across strikes, visualizes the skew, and constructs a delta-hedged risk reversal trade by selling rich downside volatility, buying upside convexity, and hedging the net option delta with SPY shares.

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

## Project Structure

```text
spy-volatility-skew-monitor/
│
├── src/
│   ├── black_scholes.py      # Black-Scholes call and put pricing
│   ├── greeks.py             # Delta, gamma, and Vega calculations
│   ├── implied_vol.py        # Newton-Raphson implied volatility solver
│   ├── price_data.py         # Historical SPY price data and log returns
│   ├── realized_vol.py       # Rolling realized volatility estimator
│   ├── skew_strategy.py      # Skew metrics and risk reversal construction
│   ├── vol_surface.py        # Live volatility skew builder and plotter
│   └── main.py               # Main project entry point
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Methodology

### 1. Market Price from Bid/Ask Midpoint

For each option, the market price is estimated using the midpoint between the bid and ask:

```python
market_price = (bid + ask) / 2
```

This midpoint is treated as the observed market option price.

### 2. Black-Scholes Theoretical Price

The project uses Black-Scholes to calculate the theoretical value of European call and put options.

The implied volatility solver repeatedly compares:

```text
Black-Scholes theoretical price - Market midpoint price
```

The goal is to find the volatility input that makes the theoretical price match the market price.

### 3. Newton-Raphson Implied Volatility Solver

Implied volatility is solved using Newton-Raphson:

```text
sigma_next = sigma - (theoretical_price - market_price) / Vega
```

Vega is used because it measures how much the option price changes with respect to volatility.

### 4. Realized Volatility Initial Guess

The solver uses recent rolling realized volatility as the initial guess for implied volatility. This helps Newton-Raphson start from a realistic volatility estimate instead of an arbitrary number.

### 5. Volatility Skew Construction

The project filters the options chain to focus on out-of-the-money options:

```text
OTM puts: strike < spot price
OTM calls: strike > spot price
```

It then calculates implied volatility for each contract and plots implied volatility against strike price.

### 6. Delta-Hedged Risk Reversal

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

## Example Output

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

## Important Note on Trade Execution

This project does not execute live trades through a brokerage API. It constructs and analyzes the trade, including the option legs and required delta hedge. The output is intended for research, learning, and options strategy analysis.

## Limitations

This project focuses on live options-chain analytics and trade construction. It does not include a full historical PnL backtest because free data sources such as `yfinance` do not provide reliable historical options chain data. A true historical backtest would require historical bid/ask quotes, implied volatilities, Greeks, and prices for the same option contracts from entry to exit.

The code is structured so that historical options data could be added later if available through a paid or institutional data source.

## Future Improvements

Potential extensions include:

* Add daily skew snapshot logging
* Build a custom historical skew dataset over time
* Compare calculated IV against Yahoo-provided IV
* Add term structure analysis across expirations
* Add interpolation or smoothing across strikes
* Add transaction cost and slippage modeling
* Add dynamic delta hedging and rebalance logic
* Integrate historical options data for true PnL backtesting
* Build a dashboard for live skew monitoring

## Technologies Used

* Python
* NumPy
* pandas
* SciPy
* matplotlib
* yfinance

## Author

Christopher Munroe  
University of Michigan  
Mathematics of Finance and Risk Management  
[LinkedIn](https://www.linkedin.com/in/chrismunroe12)