import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

FOREX_FILE = (
    RAW_DATA_DIR /
    "forex_price_data.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "forex_technical_indicators.csv"
)


# ============================================================
# SIMPLE MOVING AVERAGE
# ============================================================

def calculate_sma(df, period):

    return (
        df["Close"]
        .rolling(window=period)
        .mean()
    )


# ============================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================

def calculate_ema(df, period):

    return (
        df["Close"]
        .ewm(
            span=period,
            adjust=False
        )
        .mean()
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(df, period=14):

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=period
    ).mean()

    avg_loss = loss.rolling(
        window=period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = (
        100 -
        (100 / (1 + rs))
    )

    return rsi


# ============================================================
# MACD
# ============================================================

def calculate_macd(df):

    ema_12 = calculate_ema(
        df,
        12
    )

    ema_26 = calculate_ema(
        df,
        26
    )

    macd = ema_12 - ema_26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = (
        macd -
        signal
    )

    return (
        macd,
        signal,
        histogram
    )


# ============================================================
# BOLLINGER BANDS
# ============================================================

def calculate_bollinger_bands(
    df,
    period=20
):

    sma = (
        df["Close"]
        .rolling(window=period)
        .mean()
    )

    std = (
        df["Close"]
        .rolling(window=period)
        .std()
    )

    upper_band = (
        sma +
        (2 * std)
    )

    lower_band = (
        sma -
        (2 * std)
    )

    return (
        upper_band,
        sma,
        lower_band
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14
):

    high_low = (
        df["High"] -
        df["Low"]
    )

    high_close = np.abs(
        df["High"] -
        df["Close"].shift()
    )

    low_close = np.abs(
        df["Low"] -
        df["Close"].shift()
    )

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(window=period)
        .mean()
    )

    return atr


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "FOREX TECHNICAL INDICATOR ANALYSIS"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        FOREX_FILE
    )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"]
    )

    df = df.sort_values(
        "DateTime"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    print(
        "\nCalculating SMA..."
    )

    df["SMA_20"] = calculate_sma(
        df,
        20
    )

    df["SMA_50"] = calculate_sma(
        df,
        50
    )


    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    print(
        "Calculating EMA..."
    )

    df["EMA_12"] = calculate_ema(
        df,
        12
    )

    df["EMA_26"] = calculate_ema(
        df,
        26
    )


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    print(
        "Calculating RSI..."
    )

    df["RSI_14"] = calculate_rsi(
        df,
        14
    )


    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    print(
        "Calculating MACD..."
    )

    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Histogram"]
    ) = calculate_macd(
        df
    )


    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    print(
        "Calculating Bollinger Bands..."
    )

    (
        df["BB_Upper"],
        df["BB_Middle"],
        df["BB_Lower"]
    ) = calculate_bollinger_bands(
        df
    )


    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    print(
        "Calculating ATR..."
    )

    df["ATR_14"] = calculate_atr(
        df,
        14
    )


    # --------------------------------------------------------
    # REMOVE INITIAL NULL VALUES
    # --------------------------------------------------------

    rows_before = len(df)

    df = df.dropna().reset_index(
        drop=True
    )

    rows_after = len(df)


    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TECHNICAL INDICATOR SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nRows before processing: "
        f"{rows_before}"
    )

    print(
        f"Rows after processing: "
        f"{rows_after}"
    )

    print(
        "\nTechnical Indicators Created:"
    )

    indicators = [
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

    for indicator in indicators:

        print(
            f"✓ {indicator}"
        )


    print(
        "\nSample Data:"
    )

    print(
        df[
            [
                "DateTime",
                "PairName",
                "Close",
                "SMA_20",
                "SMA_50",
                "RSI_14",
                "MACD",
                "ATR_14"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    print("\n" + "=" * 70)

    print(
        "TECHNICAL INDICATORS COMPLETED SUCCESSFULLY"
    )

    print(
        f"Output saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
