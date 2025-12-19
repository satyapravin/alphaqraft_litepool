# RLTrader LitePool

A high-performance reinforcement learning environment for training market-making agents on cryptocurrency order book data. Built on a lightweight C++ environment pool for seamless integration with Python RL libraries.

## Overview

This project implements an **RL-based market maker** that learns to quote bid/ask spreads on BTC/USDT order book data. The agent uses **Proximal Policy Optimization (PPO)** to learn optimal quoting strategies that balance:

- **Spread capture**: Earning the bid-ask spread on completed round-trips
- **Inventory risk**: Managing position exposure to adverse price movements  
- **Fee rebates**: Earning maker rebates for providing liquidity

## Architecture

### C++ Components (`litepool/rltrader/`)

| Component | Description |
|-----------|-------------|
| `sim_exchange` | Simulated exchange with realistic order matching |
| `position` | Position tracking with P&L, fees, and leverage calculation |
| `strategy` | Avellaneda-Stoikov inspired quoting model with RL-controlled parameters |
| `env_adaptor` | Bridges market data, strategy, and exchange into RL observations |
| `market_signal_builder` | Generates 13 normalized market microstructure signals |
| `trade_reader` | Reads and synchronizes trade feed data from CSV files |
| `trade_signal_builder` | Generates 8 normalized trade flow signals (volume, pressure, intensity) |
| `amm_simulator` | Simulates AMM V3 concentrated liquidity for flow signals |
| `rltrader_litepool` | Main environment exposing Gymnasium interface to Python |

### Python Components (`litepool/rltrader/python/`)

| Component | Description |
|-----------|-------------|
| `tianshou_ppo.py` | Main training script with PPO implementation |
| `simple_actor_critic.py` | LSTM Actor-Critic with temporal pattern recognition |
| `simple_collector.py` | Experience collection from vectorized environments |
| `simple_ppo_policy.py` | PPO policy with clipped surrogate objective |

## Action Space (4 dimensions)

| Action | Range | Description |
|--------|-------|-------------|
| `bid_spread` | [-1, 1] | Controls bid quote width (mapped to [0.2x, 3.0x] spread multiplier) |
| `ask_spread` | [-1, 1] | Controls ask quote width (mapped to [0.2x, 3.0x] spread multiplier) |
| `target_inventory` | [-1, 1] | Desired inventory level (skew computed automatically from error) |
| `requote` | {0, 1} | Binary decision to cancel and requote |

**Note**: The `skew` action was removed to avoid conflicts with `target_inventory`. Quote asymmetry (skew) is now computed automatically based on the difference between current position and target inventory.

## Observation Space (26 signals)

**Market Signals (13):**
- Spread metrics: `market_spread`, `bid_depth`, `ask_depth`, `depth_imbalance`
- Volume signals: `volume_imbalance`, `volume_imbalance_trend`, `ofi` (order flow imbalance)
- Volatility: `vol_regime`, `price_trend`
- Microstructure: `spread_change`, `depth_change`

**AMM Flow Signals (4):**
- `net_flow`: EMA-based flow momentum indicator
- `flow_imbalance`: Recent buy/sell volume ratio
- `inventory_delta`: LP inventory change in simulated AMM
- `cumulative_flow/balance`: Raw cumulative flow normalized by trading balance (trend indicator)

**Trade Feed Signals (8):**
- `buy_volume`: Normalized buy volume from recent trades
- `sell_volume`: Normalized sell volume from recent trades
- `volume_imbalance`: (buy - sell) / (buy + sell) from trade flow
- `trade_intensity`: Volume per 100ms period (activity rate)
- `price_impact`: Price change per unit volume
- `buy_pressure`: EMA-based buy pressure indicator
- `sell_pressure`: EMA-based sell pressure indicator
- `time_since_last_trade`: Normalized temporal signal for trade recency

**Agent State (1):**
- `current_leverage`: Agent's current position leverage (critical for inventory management)

## Reward Function

```
reward = realized_pnl_delta + unrealized_pnl_delta + fee_rebate_delta - inventory_deviation_penalty
```

- **Realized P&L Delta**: Profit/loss from completed trades
- **Unrealized P&L Delta**: Mark-to-market changes (anchored to slow-moving price MA)
- **Fee Rebates**: Maker fee rebates earned (incentivizes market participation)
- **Inventory Deviation Penalty**: Penalizes deviation from agent's chosen `target_inventory` (aligns incentives - agent can hold positions it explicitly chooses)

## Model Architecture

**LSTM Actor-Critic** with temporal pattern recognition:

```
Observations [26] → MLP Feature Extractor [128] → LSTM [64] → Combined [192]
                                                              ↓
                                                     ┌───────┴───────┐
                                                     ↓               ↓
                                                  Actor           Critic
                                            (quote params,       (value)
                                              requote)
```

Key features:
- **MLP Feature Extractor**: 2-layer with LayerNorm for stable training
- **LSTM**: Captures temporal patterns in market dynamics
- **Exponential spread mapping**: More control at tighter spreads where profitability matters most
- **Separate actor/critic heads**: Reduces interference in learning

## Build Instructions

### Prerequisites
- CMake 3.14+
- C++17 compiler
- Python 3.8+
- PyTorch

### Build Steps

```bash
# 1. Build C++ components
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 2. Copy shared library
cp build/lib/rltrader_litepool.so litepool/rltrader/

# 3. Install Python package
pip install -e .
```

### Quick Rebuild (after C++ changes)

```bash
cd build && make -j$(nproc)
cp lib/rltrader_litepool.so ../litepool/rltrader/
```

## Training

```bash
cd litepool/rltrader/python
python tianshou_ppo.py
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_ENVS` | 3 | Parallel environments |
| `N_STEPS` | 2048 | Steps per rollout |
| `GAMMA` | 0.99 | Discount factor (~100 step horizon) |
| `BASE_SPREAD_BPS` | 3.0 | Base spread in basis points |
| `MIN_SIZE_PCT` | 5.0% | Order size as % of balance |
| `Model Params` | ~82k | LSTM Actor-Critic parameters |
| `OBS_DIM` | 26 | Observation space (13 market + 4 AMM + 8 trade + 1 agent state) |

## Data Format

Training data requires two types of CSV files:

**Order Book Data:**
```
timestamp,bid_price_1,bid_size_1,...,ask_price_1,ask_size_1,...
```
Place book data files in `data/training/` directory.

**Trade Feed Data (optional but recommended):**
```
exchange,symbol,timestamp,local_timestamp,id,side,price,amount
```
Place trade data files in `data/training/trades/` directory. The trade feed provides additional signals for better market flow understanding. If not provided, trade signals will be zeroed out.

## Configuration

Key strategy parameters in `tianshou_ppo.py`:

```python
BASE_SPREAD_BPS = 3.0    # Base spread (bps) - room for spread capture
MIN_SIZE_PCT = 5.0       # Order size (%) - strong P&L signal
MAKER_FEE = -0.000025    # Maker rebate (-0.25 bps)
TAKER_FEE = 0.0005       # Taker fee (5 bps)
```

## License

MIT License
