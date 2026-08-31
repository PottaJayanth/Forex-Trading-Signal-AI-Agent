import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\Desktop\Forex-Trading-Signal-AI-Agent"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forex_technical_indicators.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TECHNICAL FEATURES
# ============================================================

FEATURES = [
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "SMA_20",
    "SMA_50",
    "EMA_12",
    "EMA_26",
    "ATR_14"
]


# ============================================================
# FOREX PPO ENVIRONMENT
# ============================================================

class ForexTradingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, data):

        super().__init__()

        self.data = data.reset_index(drop=True)

        self.features = self.data[FEATURES].values.astype(
            np.float32
        )

        self.close_prices = self.data["Close"].values.astype(
            np.float32
        )

        self.current_step = 0

        # Actions:
        # 0 = SELL
        # 1 = HOLD
        # 2 = BUY
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(FEATURES),),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        observation = self.features[self.current_step]

        return observation, {}

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action):

        current_price = self.close_prices[
            self.current_step
        ]

        next_price = self.close_prices[
            self.current_step + 1
        ]

        price_return = (
            next_price - current_price
        ) / current_price

        # Convert action to position
        if action == 0:
            position = -1       # SELL

        elif action == 1:
            position = 0        # HOLD

        else:
            position = 1        # BUY

        # ----------------------------------------------------
        # REWARD DESIGN
        # ----------------------------------------------------

        # Reward for correctly predicting direction
        if position == 1:
            reward = price_return * 1000

        elif position == -1:
            reward = -price_return * 1000

        else:
            # HOLD receives a small penalty so the agent
            # does not automatically prefer HOLD forever.
            reward = -abs(price_return) * 100

        # Trading cost
        if position != 0:
            reward -= 0.05

        self.current_step += 1

        terminated = (
            self.current_step >= len(self.data) - 2
        )

        truncated = False

        observation = self.features[
            self.current_step
        ]

        info = {
            "price_return": float(price_return),
            "position": position,
            "reward": float(reward)
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FOREX PPO REINFORCEMENT LEARNING AGENT - VERSION 2")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading technical indicator data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Records Loaded: {len(df)}")

    df["DateTime"] = pd.to_datetime(
        df["DateTime"]
    )

    df = df.dropna(
        subset=FEATURES + ["Close"]
    ).reset_index(drop=True)

    print(
        f"Records After Cleaning: {len(df)}"
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.80
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    print(
        f"\nTraining Records: {len(train_df)}"
    )

    print(
        f"Testing Records: {len(test_df)}"
    )

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    print(
        "\nCreating Forex RL environment..."
    )

    train_env = ForexTradingEnv(
        train_df
    )

    # --------------------------------------------------------
    # TRAIN PPO
    # --------------------------------------------------------

    print(
        "\nTraining PPO agent..."
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=0.0003,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=42
    )

    model.learn(
        total_timesteps=20000
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "forex_ppo_agent_v2"
    )

    model.save(
        model_path
    )

    print(
        "\nPPO model saved to:"
    )

    print(
        model_path
    )

    # --------------------------------------------------------
    # TEST PPO
    # --------------------------------------------------------

    print(
        "\nEvaluating PPO agent on test data..."
    )

    test_env = ForexTradingEnv(
        test_df
    )

    obs, _ = test_env.reset()

    results = []

    portfolio_value = 100000.0

    initial_value = portfolio_value

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = int(action)

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info
        ) = test_env.step(action)

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        if action == 0:
            signal = "SELL"

        elif action == 1:
            signal = "HOLD"

        else:
            signal = "BUY"

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------

        pnl = (
            reward * portfolio_value / 1000
        )

        portfolio_value += pnl

        # Prevent unrealistic negative portfolio
        portfolio_value = max(
            portfolio_value,
            0
        )

        current_index = min(
            test_env.current_step,
            len(test_df) - 1
        )

        results.append({

            "DateTime":
                test_df.iloc[
                    current_index
                ]["DateTime"],

            "Close":
                test_df.iloc[
                    current_index
                ]["Close"],

            "PPO_Action":
                action,

            "PPO_Signal":
                signal,

            "PriceReturn":
                info["price_return"],

            "Reward":
                reward,

            "PnL":
                pnl,

            "PortfolioValue":
                portfolio_value
        })

        obs = next_obs

        if terminated or truncated:
            break

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "ppo_signal_predictions_v2.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    total_pnl = (
        portfolio_value - initial_value
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PPO V2 PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal Test Decisions: "
        f"{len(results_df)}"
    )

    print(
        "\nPPO Signal Distribution:"
    )

    print(
        results_df[
            "PPO_Signal"
        ].value_counts()
    )

    print(
        f"\nInitial Portfolio Value: "
        f"{initial_value:.2f}"
    )

    print(
        f"Final Portfolio Value: "
        f"{portfolio_value:.2f}"
    )

    print(
        f"Total PPO PnL: "
        f"{total_pnl:.2f}"
    )

    print(
        "\nSample PPO Decisions:"
    )

    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    print(
        "\nPPO predictions saved to:"
    )

    print(
        output_file
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PPO REINFORCEMENT LEARNING V2 COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
