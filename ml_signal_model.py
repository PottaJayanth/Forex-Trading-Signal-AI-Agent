import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "forex_signals.csv"
)

PREDICTIONS_FILE = (
    PROCESSED_DATA_DIR /
    "ml_signal_predictions.csv"
)

IMPORTANCE_FILE = (
    PROCESSED_DATA_DIR /
    "ml_feature_importance.csv"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("MACHINE LEARNING FOREX SIGNAL MODEL")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nRecords Loaded: {len(df)}"
    )


    # --------------------------------------------------------
    # CREATE TARGET
    # --------------------------------------------------------

    df["NextClose"] = df["Close"].shift(-1)

    df = df.dropna(
        subset=["NextClose"]
    ).copy()


    # 1 = Price goes UP
    # 0 = Price goes DOWN or stays same

    df["Target"] = (
        df["NextClose"] >
        df["Close"]
    ).astype(int)


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [

        "SMA_20",
        "SMA_50",

        "EMA_12",
        "EMA_26",

        "RSI_14",

        "MACD",
        "MACD_Signal",
        "MACD_Histogram",

        "BB_Upper",
        "BB_Middle",
        "BB_Lower",

        "ATR_14"
    ]


    X = df[features]

    y = df["Target"]


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    print(
        f"\nTraining Records: "
        f"{len(X_train)}"
    )

    print(
        f"Testing Records: "
        f"{len(X_test)}"
    )


    # --------------------------------------------------------
    # BUILD RANDOM FOREST MODEL
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest model..."
    )

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=8,

        random_state=42,

        class_weight="balanced"
    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance":
            model.feature_importances_

    })


    importance_df = (

        importance_df

        .sort_values(

            "Importance",

            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance_df
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # TRAIN MODEL ON FULL DATA
    # --------------------------------------------------------

    model.fit(
        X,
        y
    )


    # Predict all records

    df["ML_Prediction"] = model.predict(
        X
    )


    df["ML_Probability_Up"] = (

        model

        .predict_proba(X)

        [:, 1]
    )


    # Convert prediction to label

    df["ML_Signal"] = df[
        "ML_Prediction"
    ].map({

        1: "BUY",

        0: "SELL"
    })


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    output_columns = [

        "DateTime",

        "PairName",

        "Close",

        "RSI_14",

        "MACD",

        "ATR_14",

        "TradingSignal",

        "SignalScore",

        "Target",

        "ML_Prediction",

        "ML_Probability_Up",

        "ML_Signal"
    ]


    df[
        output_columns
    ].to_csv(

        PREDICTIONS_FILE,

        index=False
    )


    # --------------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df.to_csv(

        IMPORTANCE_FILE,

        index=False
    )


    # --------------------------------------------------------
    # SAMPLE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE ML PREDICTIONS")
    print("=" * 70)

    print(

        df[
            output_columns
        ]

        .head(15)

        .to_string(
            index=False
        )
    )


    print("\n" + "=" * 70)
    print(
        "MACHINE LEARNING MODEL COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        f"\nPredictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{IMPORTANCE_FILE}"
    )


if __name__ == "__main__":
    main()
