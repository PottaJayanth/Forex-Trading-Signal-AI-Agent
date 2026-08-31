import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

INPUT_FILE = (
    RAW_DATA_DIR /
    "mc_scenarios.csv"
)

VAR_OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "var_cvar_results.csv"
)

WORST_SCENARIOS_FILE = (
    PROCESSED_DATA_DIR /
    "worst_monte_carlo_scenarios.csv"
)


# ============================================================
# CALCULATE VAR AND CVAR
# ============================================================

def calculate_var_cvar(df, confidence):

    # VaR percentile
    var_value = df["PortfolioPnL"].quantile(
        1 - confidence
    )

    # Loss expressed as positive value
    var_loss = abs(var_value)

    # Tail scenarios beyond VaR
    tail_losses = df[
        df["PortfolioPnL"] <= var_value
    ]

    # CVaR
    cvar_value = abs(
        tail_losses[
            "PortfolioPnL"
        ].mean()
    )

    return (
        var_loss,
        cvar_value,
        len(tail_losses)
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FOREX MONTE CARLO RISK ANALYSIS")
    print("=" * 70)


    # Load scenarios

    df = pd.read_csv(INPUT_FILE)
    
    print("COLUMN NAMES:")
    print(df.columns.tolist())

    print(
        f"\nTotal Scenarios: {len(df)}"
    )


    print("\nPortfolio PnL Summary:")

    print(
        df["PortfolioPnL"]
        .describe()
    )


    # ========================================================
    # VaR AND CVaR
    # ========================================================

    results = []

    for confidence in [0.95, 0.99]:

        var_value, cvar_value, tail_count = (
            calculate_var_cvar(
                df,
                confidence
            )
        )

        results.append({

            "ConfidenceLevel":
                f"{int(confidence * 100)}%",

            "VaR":
                var_value,

            "CVaR":
                cvar_value,

            "TailScenarios":
                tail_count
        })


    results_df = pd.DataFrame(
        results
    )


    print("\n" + "=" * 70)
    print("VaR AND CVaR RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # WORST SCENARIOS
    # ========================================================

    worst_scenarios = (

        df

        .sort_values(
            "PortfolioPnL"
        )

        .head(10)

        .copy()
    )


    print("\n" + "=" * 70)
    print("WORST 10 MONTE CARLO SCENARIOS")
    print("=" * 70)

    print(
        worst_scenarios.to_string(
            index=False
        )
    )


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    worst_loss = abs(
        df["PortfolioPnL"].min()
    )


    if worst_loss > 200000:

        risk_level = "HIGH"

    elif worst_loss > 100000:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    print("\n" + "=" * 70)
    print("PORTFOLIO RISK CLASSIFICATION")
    print("=" * 70)

    print(
        f"Risk Level: {risk_level}"
    )

    print(
        f"Worst Simulated Loss: "
        f"{worst_loss:,.2f}"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        VAR_OUTPUT_FILE,
        index=False
    )

    worst_scenarios.to_csv(
        WORST_SCENARIOS_FILE,
        index=False
    )


    print("\n" + "=" * 70)
    print(
        "MONTE CARLO RISK ANALYSIS COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        f"\nVaR/CVaR results saved to:\n"
        f"{VAR_OUTPUT_FILE}"
    )

    print(
        f"\nWorst scenarios saved to:\n"
        f"{WORST_SCENARIOS_FILE}"
    )


if __name__ == "__main__":
    main()
