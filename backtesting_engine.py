import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "forex_signals.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "backtest_results.csv"
)

SUMMARY_FILE = (
    PROCESSED_DATA_DIR /
    "backtest_summary.csv"
)


# ============================================================
# BACKTEST SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

POSITION_SIZE = 10000

TRANSACTION_COST_RATE = 0.0001


# ============================================================
# PREPARE TRADING DATA
# ============================================================

def prepare_backtest_data(df):

    # Next period close price
    df["NextClose"] = df["Close"].shift(-1)

    # Remove final row because it has no next price
    df = df.dropna(
        subset=["NextClose"]
    ).copy()

    # Direction
    df["Direction"] = 0

    df.loc[
        df["TradingSignal"].isin(
            ["BUY", "STRONG_BUY"]
        ),
        "Direction"
    ] = 1

    df.loc[
        df["TradingSignal"].isin(
            ["SELL", "STRONG_SELL"]
        ),
        "Direction"
    ] = -1

    return df


# ============================================================
# CALCULATE PNL
# ============================================================

def calculate_pnl(df):

    # Price movement
    df["PriceChange"] = (
        df["NextClose"] -
        df["Close"]
    )

    # Percentage return
    df["MarketReturn"] = (
        df["NextClose"] -
        df["Close"]
    ) / df["Close"]

    # Gross PnL
    df["GrossPnL"] = (
        df["Direction"] *
        df["MarketReturn"] *
        POSITION_SIZE
    )

    # Transaction costs only when a trade occurs
    df["TransactionCost"] = np.where(
        df["Direction"] != 0,
        POSITION_SIZE *
        TRANSACTION_COST_RATE,
        0
    )

    # Net PnL
    df["NetPnL"] = (
        df["GrossPnL"] -
        df["TransactionCost"]
    )

    return df


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(df):

    trades = df[
        df["Direction"] != 0
    ].copy()

    total_trades = len(trades)

    winning_trades = len(
        trades[
            trades["NetPnL"] > 0
        ]
    )

    losing_trades = len(
        trades[
            trades["NetPnL"] < 0
        ]
    )

    win_rate = (
        winning_trades /
        total_trades * 100
        if total_trades > 0
        else 0
    )

    gross_profit = trades.loc[
        trades["NetPnL"] > 0,
        "NetPnL"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["NetPnL"] < 0,
            "NetPnL"
        ].sum()
    )

    profit_factor = (
        gross_profit /
        gross_loss
        if gross_loss != 0
        else 0
    )

    # Portfolio equity curve
    df["CumulativePnL"] = (
        df["NetPnL"].cumsum()
    )

    df["PortfolioValue"] = (
        INITIAL_CAPITAL +
        df["CumulativePnL"]
    )

    # Running maximum
    df["RunningMax"] = (
        df["PortfolioValue"].cummax()
    )

    # Drawdown percentage
    df["DrawdownPct"] = (
        (
            df["PortfolioValue"] -
            df["RunningMax"]
        ) /
        df["RunningMax"]
    ) * 100

    max_drawdown = (
        df["DrawdownPct"].min()
    )

    # Strategy returns for Sharpe Ratio
    strategy_returns = (
        df["NetPnL"] /
        INITIAL_CAPITAL
    )

    if strategy_returns.std() != 0:

        sharpe_ratio = (
            strategy_returns.mean() /
            strategy_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe_ratio = 0


    total_pnl = (
        df["NetPnL"].sum()
    )

    final_portfolio_value = (
        INITIAL_CAPITAL +
        total_pnl
    )


    metrics = {
        "InitialCapital": INITIAL_CAPITAL,
        "FinalPortfolioValue": final_portfolio_value,
        "TotalPnL": total_pnl,
        "TotalTrades": total_trades,
        "WinningTrades": winning_trades,
        "LosingTrades": losing_trades,
        "WinRatePct": win_rate,
        "ProfitFactor": profit_factor,
        "SharpeRatio": sharpe_ratio,
        "MaxDrawdownPct": max_drawdown
    }

    return metrics, df


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FOREX TRADING STRATEGY BACKTEST")
    print("=" * 70)


    # Load data
    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nRecords Loaded: {len(df)}"
    )


    # Prepare data
    df = prepare_backtest_data(df)


    # Calculate PnL
    df = calculate_pnl(df)


    # Calculate metrics
    metrics, df = calculate_metrics(df)


    # Save detailed results
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # Save summary
    summary_df = pd.DataFrame(
        list(metrics.items()),
        columns=[
            "Metric",
            "Value"
        ]
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("BACKTEST PERFORMANCE RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{metric}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{metric}: "
                f"{value}"
            )


    # Sample trades

    print("\n" + "=" * 70)
    print("SAMPLE BACKTEST RESULTS")
    print("=" * 70)

    sample_columns = [
        "DateTime",
        "PairName",
        "Close",
        "NextClose",
        "TradingSignal",
        "Direction",
        "GrossPnL",
        "TransactionCost",
        "NetPnL",
        "PortfolioValue",
        "DrawdownPct"
    ]

    print(
        df[
            sample_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


    print("\n" + "=" * 70)
    print(
        "BACKTESTING COMPLETED SUCCESSFULLY"
    )

    print(
        f"\nDetailed results:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nPerformance summary:\n"
        f"{SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
