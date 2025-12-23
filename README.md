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
| `bid_spread` | [-1, 1] | Controls bid quote width (mapped to [0.5x, 50x] spread multiplier) |
| `ask_spread` | [-1, 1] | Controls ask quote width (mapped to [0.5x, 50x] spread multiplier) |
| `target_inventory` | [-1, 1] | Desired inventory level (scaled to ±10% leverage, skew computed automatically) |
| `requote` | {0, 1} | Binary decision to cancel and requote |

**Note**: The `skew` action was removed to avoid conflicts with `target_inventory`. Quote asymmetry (skew) is now computed automatically based on the difference between current position and target inventory.

## Observation Space (31 signals)

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

**Agent State (7):**
- `current_leverage`: Agent's current position leverage (critical for inventory management)
- `normalized_position`: Position value normalized by initial balance (direct inventory signal)
- `normalized_unrealized_pnl`: Unrealized P&L normalized by initial balance (mark-to-market performance)
- `normalized_realized_pnl`: Realized P&L normalized by initial balance (locked-in performance)
- `normalized_spread_capture`: Spread capture normalized by initial balance (direct reward signal)
- `deviation_from_target`: Signed deviation of current leverage from target leverage (positive = over-leveraged, negative = under-leveraged)
- `unrealized_pnl_pct`: Unrealized P&L as % of position value (±1% maps to ±0.76, helps agent learn to close at profit/loss thresholds)

## Reward Function

```
reward = (1.0 * realized_pnl_delta + 10.0 * spread_capture_delta)
       + 1.0 * unrealized_pnl_delta
       + 10.0 * fee_rebate_delta 
       + requote_penalty
```

All deltas are normalized by initial balance to make rewards scale-independent. Final reward is scaled by 10,000 for readability.

### Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Realized P&L Delta** | 1.0 | Profit/loss from completed trades (average price accounting) |
| **Spread Capture Delta** | 10.0 | Profit from round-trips (LIFO, boosted to match unrealized P&L scale) |
| **Unrealized P&L Delta** | 1.0 | Mark-to-market changes (aligns reward with actual P&L) |
| **Fee Rebate Delta** | 10.0 | Maker rebates earned (strongly incentivizes trading activity) |
| **Requote Penalty** | -0.0001 | Per voluntary requote (normalized, encourages order persistence) |

### Design Notes

- **Spread capture boosted (10x)**: Spread capture is ~50x smaller scale than unrealized P&L, so needs higher weight to incentivize round-trips
- **Symmetric unrealized P&L (1x)**: Aligns reward with actual P&L (asymmetric weighting caused systematic negative bias due to price oscillations)
- **Fee weight (10x)**: Strongly incentivizes trading activity (agent learned to not trade when weight was too low)
- **Requote penalty**: Only penalizes *voluntary* requotes (agent choice), not forced requotes (first step, no orders, after fills)

## Model Architecture

**LSTM Actor-Critic** with temporal pattern recognition:

```
Observations [31] → MLP Feature Extractor [128] → LSTM [64] → Combined [192]
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
| `NUM_ENVS` | 6 | Parallel environments |
| `N_STEPS` | 2048 | Steps per rollout |
| `GAMMA` | 0.995 | Discount factor |
| `BASE_SPREAD_BPS` | 1.0 | Base spread in basis points |
| `MIN_SIZE_PCT` | 1.0% | Order size as % of balance (per level, 5 levels = 5% total per side) |
| `Model Params` | ~84k | LSTM Actor-Critic parameters |
| `OBS_DIM` | 32 | Observation space (13 market + 4 AMM + 8 trade + 7 agent state) |

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
BASE_SPREAD_BPS = 1.0    # Base spread (bps) - room for spread capture
MIN_SIZE_PCT = 1.0       # Order size (%) per level (5 levels = 5% total per side)
MAKER_FEE = -0.000025    # Maker rebate (-0.25 bps)
TAKER_FEE = 0.0005       # Taker fee (5 bps)
```

## Info Fields

The environment exposes several info fields for monitoring and debugging:

- `deviation_from_target`: Signed deviation of current leverage from target leverage
  - Positive when leverage > target (over-leveraged)
  - Negative when leverage < target (under-leveraged)
  - Useful for tracking how well the agent maintains its target position

- `unrealized_pnl_pct`: Unrealized P&L as percentage of position value
  - Tells the agent "I'm up/down X% on my current position"
  - ±1% P/L maps to ±0.76 (tanh scaling with 100x multiplier)
  - Agent can learn to close positions at profit/loss thresholds (e.g., ±1%)

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
