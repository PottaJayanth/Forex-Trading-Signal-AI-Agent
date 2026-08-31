import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "ml_signal_predictions.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "ai_agent_signals.csv"
)


# ============================================================
# AI AGENT DECISION FUNCTION
# ============================================================

def generate_ai_decision(row):

    rule_signal = row["TradingSignal"]

    ml_signal = row["ML_Signal"]

    probability = row["ML_Probability_Up"]

    rsi = row["RSI_14"]

    atr = row["ATR_14"]


    # --------------------------------------------------------
    # DETERMINE CONFIDENCE
    # --------------------------------------------------------

    if probability >= 0.75 or probability <= 0.25:
        confidence = "HIGH"

    elif probability >= 0.60 or probability <= 0.40:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"


    # --------------------------------------------------------
    # DETERMINE RISK LEVEL
    # --------------------------------------------------------

    if atr > 0.002:
        risk_level = "HIGH"

    elif atr > 0.001:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"


    # --------------------------------------------------------
    # AI DECISION
    # --------------------------------------------------------

    if (
        rule_signal in ["BUY", "STRONG_BUY"]
        and ml_signal == "BUY"
        and probability >= 0.60
    ):

        decision = "BUY"


    elif (
        rule_signal in ["SELL", "STRONG_SELL"]
        and ml_signal == "SELL"
        and probability <= 0.40
    ):

        decision = "SELL"


    else:

        decision = "HOLD"


    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    if risk_level == "HIGH":

        if decision in ["BUY", "SELL"]:

            decision = "HOLD"


    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------

    if decision == "BUY":

        recommendation = (
            "Bullish technical and ML signals agree. "
            "Consider a controlled long position."
        )

    elif decision == "SELL":

        recommendation = (
            "Bearish technical and ML signals agree. "
            "Consider a controlled short position."
        )

    else:

        recommendation = (
            "Signals are not sufficiently aligned. "
            "Wait for stronger confirmation."
        )


    return (
        decision,
        confidence,
        risk_level,
        recommendation
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FOREX AI TRADING AGENT")
    print("=" * 70)


    # Load data

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nRecords Loaded: {len(df)}"
    )


    print(
        "\nGenerating AI trading decisions..."
    )


    # Generate decisions

    results = df.apply(
        generate_ai_decision,
        axis=1
    )


    df["AI_Decision"] = [
        result[0]
        for result in results
    ]

    df["ConfidenceLevel"] = [
        result[1]
        for result in results
    ]

    df["RiskLevel"] = [
        result[2]
        for result in results
    ]

    df["Recommendation"] = [
        result[3]
        for result in results
    ]


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # DISPLAY AI DECISION SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("AI DECISION DISTRIBUTION")
    print("=" * 70)

    print(
        df["AI_Decision"]
        .value_counts()
    )


    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 70)

    print(
        df["ConfidenceLevel"]
        .value_counts()
    )


    print("\n" + "=" * 70)
    print("RISK LEVEL DISTRIBUTION")
    print("=" * 70)

    print(
        df["RiskLevel"]
        .value_counts()
    )


    # ========================================================
    # SAMPLE RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAMPLE AI AGENT DECISIONS")
    print("=" * 70)

    sample_columns = [

        "DateTime",
        "PairName",
        "Close",

        "RSI_14",
        "ATR_14",

        "TradingSignal",
        "ML_Signal",
        "ML_Probability_Up",

        "AI_Decision",
        "ConfidenceLevel",
        "RiskLevel",
        "Recommendation"
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
        "AI TRADING AGENT COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        f"\nOutput saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
