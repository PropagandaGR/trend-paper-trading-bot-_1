# Trend Paper Trading Bot

A paper-trading project for experimenting with trading strategies using simulated money only.

## Safety and scope

- Uses simulated funds for testing and learning.
- Does not place real trades.
- Does not connect to a brokerage account.
- Do not store broker credentials, API keys, or other secrets in this repository.

## Run in GitHub Codespaces

1. Open the [repository](https://github.com/PropagandaGR/trend-paper-trading-bot-_1).
2. Select **Code → Codespaces → Create codespace on main**. If a Codespaces environment already exists, open it instead.
3. Wait for setup to finish; the development container installs dependencies from `requirements.txt`.
4. In the Codespaces terminal, run:

   ```bash
   python -m py_compile paper_trader.py
   python paper_trader.py
   ```

5. Inspect the generated files under `results/`:
   - `summary.json`: headline performance metrics and limitations.
   - `equity_curve.csv`: daily simulated equity and returns.
   - `trades.csv`: position-change dates.

If dependency installation fails, run `python -m pip install -r requirements.txt` and retry. The script needs internet access to download historical prices from Yahoo Finance.

## Strategy and data

The initial script tests a long-only monthly trend/momentum strategy using historical ETF prices and compares it with VT. It uses simulated starting cash of $15 and estimated commission/slippage. Historical results are not predictions of future performance.

## Important limitations

- This is a historical backtest, not a live trading system.
- It places no real orders and has no broker integration.
- Data availability, ETF selection, execution-price assumptions, market holidays, taxes, and fund costs can affect results.
- Do not interpret a successful run or positive backtest as evidence of future profits.
