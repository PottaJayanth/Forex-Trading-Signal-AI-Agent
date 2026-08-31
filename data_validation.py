import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# FILE PATHS
# ============================================================

FOREX_FILE = RAW_DATA_DIR / "forex_price_data.csv"
TRADE_FILE = RAW_DATA_DIR / "trade_log.csv"
MC_FILE = RAW_DATA_DIR / "mc_scenarios.csv"


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_data(df, dataset_name):

    print("\n" + "=" * 70)
    print(f"VALIDATING: {dataset_name}")
    print("=" * 70)

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nCOLUMN NAMES:")

    for column in df.columns:
        print(f" - {column}")

    print("\nMISSING VALUES:")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing)

    print("\nDUPLICATE ROWS:")

    duplicates = df.duplicated().sum()

    print(duplicates)

    print("\nDATA TYPES:")

    print(df.dtypes)

    print("\nSAMPLE DATA:")

    print(df.head(5))

    print("\n" + "=" * 70)




def main():

    try:

        

        forex_df = pd.read_csv(FOREX_FILE)

        trade_df = pd.read_csv(TRADE_FILE)

        mc_df = pd.read_csv(MC_FILE)


        

        validate_data(
            forex_df,
            "FOREX PRICE DATA"
        )

        validate_data(
            trade_df,
            "TRADE LOG"
        )

        validate_data(
            mc_df,
            "MONTE CARLO SCENARIOS"
        )



        print("\n" + "=" * 70)
        print("FOREX DATA SUMMARY")
        print("=" * 70)

        print(
            "\nCurrency Pairs:"
        )

        print(
            forex_df[
                "PairName"
            ].value_counts()
        )

        print(
            "\nDate Range:"
        )

        print(
            forex_df[
                "DateTime"
            ].min(),
            "to",
            forex_df[
                "DateTime"
            ].max()
        )


       

        print("\n" + "=" * 70)
        print("TRADE DATA SUMMARY")
        print("=" * 70)

        print(
            "\nStrategies:"
        )

        print(
            trade_df[
                "StrategyName"
            ].value_counts()
        )

        print(
            "\nSignal Types:"
        )

        print(
            trade_df[
                "SignalType"
            ].value_counts()
        )


       

        print("\n" + "=" * 70)
        print("MONTE CARLO DATA SUMMARY")
        print("=" * 70)

        print(
            f"\nTotal Scenarios: "
            f"{len(mc_df)}"
        )

        print(
            "\nPortfolio PnL Summary:"
        )

        print(
            mc_df[
                "PortfolioPnL"
            ].describe()
        )


        

        print("\n" + "=" * 70)
        print(
            "ALL FOREX DATASETS LOADED "
            "AND VALIDATED SUCCESSFULLY"
        )
        print("=" * 70)


    except FileNotFoundError as e:

        print("\nERROR: FILE NOT FOUND")

        print(e)

        print(
            "\nCheck that all CSV files are inside:"
        )

        print(RAW_DATA_DIR)


if __name__ == "__main__":
    main()
