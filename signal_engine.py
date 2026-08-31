import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "forex_technical_indicators.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "forex_signals.csv"
)


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(row):

    score = 0

    # --------------------------------------------------------
    # 1. SMA TREND SIGNAL
    # --------------------------------------------------------

    if row["SMA_20"] > row["SMA_50"]:
        score += 1
    else:
        score -= 1


    # --------------------------------------------------------
    # 2. EMA TREND SIGNAL
    # --------------------------------------------------------

    if row["EMA_12"] > row["EMA_26"]:
        score += 1
    else:
        score -= 1


    # --------------------------------------------------------
    # 3. RSI SIGNAL
    # --------------------------------------------------------

    if row["RSI_14"] < 30:
        score += 1

    elif row["RSI_14"] > 70:
        score -= 1


    # --------------------------------------------------------
    # 4. MACD SIGNAL
    # --------------------------------------------------------

    if row["MACD"] > row["MACD_Signal"]:
        score += 1
    else:
        score -= 1


    # --------------------------------------------------------
    # 5. BOLLINGER BAND SIGNAL
    # --------------------------------------------------------

    if row["Close"] < row["BB_Lower"]:
        score += 1

    elif row["Close"] > row["BB_Upper"]:
        score -= 1


    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    if score >= 3:
        return score, "STRONG_BUY"

    elif score == 2:
        return score, "BUY"

    elif score == 1:
        return score, "HOLD"

    elif score == 0:
        return score, "HOLD"

    elif score == -1:
        return score, "HOLD"

    elif score == -2:
        return score, "SELL"

    else:
        return score, "STRONG_SELL"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FOREX TRADING SIGNAL ENGINE")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nTotal Records Loaded: "
        f"{len(df)}"
    )


    # --------------------------------------------------------
    # GENERATE SIGNALS
    # --------------------------------------------------------

    print(
        "\nGenerating trading signals..."
    )

    results = df.apply(
        generate_signal,
        axis=1
    )

    df["SignalScore"] = [
        result[0]
        for result in results
    ]

    df["TradingSignal"] = [
        result[1]
        for result in results
    ]


    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # SIGNAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIGNAL DISTRIBUTION")
    print("=" * 70)

    print(
        df["TradingSignal"]
        .value_counts()
    )

    print("\nSignal Score Distribution:")

    print(
        df["SignalScore"]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # SAMPLE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE SIGNALS")
    print("=" * 70)

    print(
        df[
            [
                "DateTime",
                "PairName",
                "Close",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "SignalScore",
                "TradingSignal"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


    print("\n" + "=" * 70)
    print("FOREX SIGNAL ENGINE COMPLETED SUCCESSFULLY")
    print(
        f"Output saved to:\n"
        f"{OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
