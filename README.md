# Forex-Trading-Signal-AI-Agent
AI-powered Forex trading signal system using technical analysis, machine learning, reinforcement learning, risk management, and Power BI.
# Forex Trading Signal AI Agent

An end-to-end Forex analytics and AI trading system combining technical analysis, machine learning, reinforcement learning, backtesting, Monte Carlo risk analysis, and Power BI visualization.

## Project Overview

This project develops an AI-assisted Forex trading signal system that analyzes market data using technical indicators, machine learning predictions, and a PPO reinforcement learning agent.

The system generates BUY, HOLD, and SELL decisions and evaluates trading performance and portfolio risk through historical backtesting and Monte Carlo analysis.

## Project Architecture

Forex Price Data
        |
        v
Technical Indicators
        |
        v
Trading Signal Engine
        |
        +--------------------+
        |                    |
        v                    v
Random Forest ML       PPO Reinforcement
Model                  Learning Agent
        |                    |
        +---------+----------+
                  |
                  v
            AI Trading Agent
                  |
          BUY / HOLD / SELL
                  |
          +-------+-------+
          |               |
          v               v
     Backtesting     Risk Analysis
                          |
                    VaR / CVaR /
                  Monte Carlo Scenarios
                          |
                          v
                      Power BI

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gymnasium
- Stable-Baselines3
- PPO Reinforcement Learning
- Power BI
- DAX
- Monte Carlo Simulation

## Project Components

### 1. Data Validation

Forex market data, trade logs, and Monte Carlo scenarios were validated for:

- Missing values
- Duplicate records
- Data types
- Dataset structure
- Date ranges

### 2. Technical Analysis

The system calculates:

- SMA 20
- SMA 50
- EMA 12
- EMA 26
- RSI 14
- MACD
- MACD Signal
- MACD Histogram
- Bollinger Bands
- ATR 14

### 3. Trading Signal Engine

Technical indicators are combined to generate:

- STRONG_BUY
- BUY
- HOLD
- SELL
- STRONG_SELL

### 4. Machine Learning

A Random Forest classification model predicts the direction of the next market movement.

Model test accuracy:

**56.67%**

Important features included:

- MACD Histogram
- RSI
- MACD
- ATR
- MACD Signal
- EMA indicators
- Bollinger Band indicators

### 5. Reinforcement Learning

A custom Forex trading environment was implemented using Gymnasium.

The PPO agent uses technical indicators as observations and selects among:

- BUY
- HOLD
- SELL

The PPO model was trained using Stable-Baselines3.

The current PPO experiment is retained as a model evaluation artifact. The test-period policy showed a strong directional bias toward BUY and produced a negative test P&L of approximately ₹488.95. Therefore, the PPO result is not presented as profitable live-trading performance.

### 6. AI Trading Agent

The AI decision layer combines technical signals and machine learning predictions to generate:

- BUY
- HOLD
- SELL

It also assigns:

- Confidence Level
- Risk Level
- Trading Recommendation

### 7. Backtesting

The trading strategy was evaluated using historical data.

Key results:

| Metric | Result |
|---|---:|
| Initial Capital | ₹100,000 |
| Final Portfolio Value | ₹99,839.22 |
| Total P&L | -₹160.78 |
| Total Trades | 181 |
| Winning Trades | 79 |
| Losing Trades | 102 |
| Win Rate | 43.65% |
| Profit Factor | 0.7867 |
| Sharpe Ratio | -0.9727 |
| Maximum Drawdown | -0.1759% |

These results represent historical backtest performance and are not a guarantee of future trading results.

### 8. Monte Carlo Risk Analysis

The project evaluates portfolio risk using 1,000 Monte Carlo scenarios.

| Risk Metric | Result |
|---|---:|
| 95% VaR | ₹165,865.70 |
| 95% CVaR | ₹198,183.87 |
| 99% VaR | ₹223,141.70 |
| 99% CVaR | ₹245,497.55 |
| Worst Simulated Loss | ₹269,801.36 |
| Risk Classification | HIGH |

### 9. Power BI Dashboard

The Power BI dashboard provides monitoring and analysis across five pages:

1. Executive Dashboard
2. Trading Signals
3. AI & ML Performance
4. Risk Management
5. Backtesting & Performance

The dashboard includes KPI cards, trading signal analysis, ML probability analysis, AI decisions, Monte Carlo risk analysis, portfolio performance, drawdown analysis, and trading performance attribution.

## Project Outputs

### Processed Data

- `forex_technical_indicators.csv`
- `forex_signals.csv`
- `backtest_results.csv`
- `backtest_summary.csv`
- `ml_signal_predictions.csv`
- `ml_feature_importance.csv`
- `ai_agent_signals.csv`
- `var_cvar_results.csv`
- `worst_monte_carlo_scenarios.csv`
- `ppo_signal_predictions.csv`
- `ppo_signal_predictions_v2.csv`

### Machine Learning / RL Models

- Random Forest signal model
- PPO reinforcement learning model
- `forex_ppo_agent.zip`
- `forex_ppo_agent_v2.zip`

## Repository Structure

```text
Forex-Trading-Signal-AI-Agent/
|
├── data/
│   ├── raw/
│   └── processed/
|
├── models/
|
├── src/
|
├── powerbi/
|
├── screenshots/
|
└── README.md
