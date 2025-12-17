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
| `amm_simulator` | Simulates AMM V3 concentrated liquidity for flow signals |
| `rltrader_litepool` | Main environment exposing Gymnasium interface to Python |

### Python Components (`litepool/rltrader/python/`)

| Component | Description |
|-----------|-------------|
| `tianshou_ppo.py` | Main training script with PPO implementation |
| `simple_actor_critic.py` | Neural network with shared features, separate actor/critic heads |
| `simple_collector.py` | Experience collection from vectorized environments |
| `simple_ppo_policy.py` | PPO policy with clipped surrogate objective |

## Action Space (5 dimensions)

| Action | Range | Description |
|--------|-------|-------------|
| `bid_spread` | [-1, 1] | Controls bid quote width (mapped to spread multiplier) |
| `ask_spread` | [-1, 1] | Controls ask quote width (mapped to spread multiplier) |
| `skew` | [-1, 1] | Asymmetric quote adjustment for inventory control |
| `target_inventory` | [-1, 1] | Desired inventory level (EMA smoothed) |
| `requote` | {0, 1} | Binary decision to cancel and requote |

## Observation Space (16 signals)

**Market Signals (13):**
- Spread metrics: `market_spread`, `bid_depth`, `ask_depth`, `depth_imbalance`
- Volume signals: `volume_imbalance`, `volume_imbalance_trend`, `ofi` (order flow imbalance)
- Volatility: `vol_regime`, `price_trend`
- Microstructure: `spread_change`, `depth_change`

**AMM Flow Signals (3):**
- `buy_pressure`: Recent buy volume from simulated AMM arbitrage
- `sell_pressure`: Recent sell volume from simulated AMM arbitrage  
- `net_flow`: Directional flow momentum indicator

## Reward Function

```
reward = realized_pnl_delta + unrealized_pnl_delta + fee_rebate_delta
```

- **Realized P&L Delta**: Profit/loss from completed trades
- **Unrealized P&L Delta**: Mark-to-market changes (anchored to slow-moving price MA)
- **Fee Rebates**: Maker fee rebates earned (incentivizes market participation)

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
| `GAMMA` | 0.995 | Discount factor (~200 step horizon) |
| `BASE_SPREAD_BPS` | 1.0 | Base spread in basis points |
| `MIN_SIZE_PCT` | 1.0% | Order size as % of balance |

## Data Format

Training data should be CSV files with order book snapshots:

```
timestamp,bid_price_1,bid_size_1,...,ask_price_1,ask_size_1,...
```

Place data files in `data/training/` directory.

## Configuration

Key strategy parameters in `tianshou_ppo.py`:

```python
BASE_SPREAD_BPS = 1.0    # Base spread (bps)
MIN_SIZE_PCT = 1.0       # Min order size (%)
MAX_SIZE_PCT = 5.0       # Max order size (%)
MAKER_FEE = -0.000025    # Maker rebate (-0.25 bps)
TAKER_FEE = 0.0005       # Taker fee (5 bps)
```

## License

MIT License
