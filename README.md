# RLTrader LitePool

A high-performance reinforcement learning environment for training market-making agents on cryptocurrency order book data. Built on a lightweight C++ environment pool with a **novel hierarchical two-agent architecture** that unifies trend-following and market-making into a single coherent system.

---

## 🚀 The Innovation: Two-Agent Hierarchical RL

Traditional market makers face an impossible choice:

| Approach | Problem |
|----------|---------|
| **Pure Market Making** | Ignores market direction → gets run over by trends |
| **Pure Trend Following** | Pays spread to enter/exit → erodes profits |
| **Single RL Agent** | Conflicting objectives → learns to game rewards |

**Our solution**: Decompose the problem into two specialized agents that cooperate:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INVENTORY AGENT (Strategic)                        │
│  "What position should I hold given market conditions?"                   │
│                                                                           │
│  • Observes: AMM flow signals, trade pressure, volatility                 │
│  • Decides: Target inventory level (±10% leverage)                        │
│  • Reward: Unrealized P&L changes (learns market direction)               │
│  • Updates: Every 100 steps (50 seconds) - strategic time scale           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ target_inventory
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      MARKET MAKING AGENT (Tactical)                       │
│  "How do I efficiently reach that target while capturing spread?"         │
│                                                                           │
│  • Observes: Microstructure signals + target from Inventory Agent         │
│  • Decides: Bid/ask spreads (smart requote handles timing)                │
│  • Reward: Spread capture + fee rebates - inactivity penalty              │
│  • Updates: Every step (500ms) - tactical time scale                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why This Works

| Benefit | Explanation |
|---------|-------------|
| **No Reward Gaming** | MM agent can't inflate rewards by only opening positions - it's paid for round-trips |
| **Smart Requoting** | Can't avoid fills by never requoting - requotes triggered automatically by market moves |
| **Clean Separation** | Inventory agent learns *when* to be long/short; MM agent learns *how* to get there |
| **Different Time Scales** | Strategic decisions (seconds) don't interfere with tactical execution (milliseconds) |
| **Interpretable** | Can analyze each agent's learned behavior independently |
| **Trend + Spread** | Captures directional moves (via inventory) while earning spread (via MM) |

---

## 📊 Reward Structure

The key insight: **each agent optimizes what it can control**.

| Agent | Reward | Controls | Signal |
|-------|--------|----------|--------|
| **Inventory** | `Δ(unrealized_pnl)` | When to be long/short/flat | Market direction |
| **MM** | `spread_capture + fees` | Quote pricing, execution timing | Execution quality |

```cpp
// MM Agent: optimizes spread capture (LIFO) + fee rebates
mm_reward = (spread_capture_delta + fee_delta) × 10,000
mm_reward -= 0.001 if no fills this step  // inactivity penalty

// Inventory Agent: optimizes unrealized P&L (market direction)
inv_reward = unrealized_pnl_delta × 10,000

// Combined reward = total wealth change
total_reward = mm_reward + inv_reward
```

**Inactivity Penalty**: The MM agent receives a small penalty (-0.001) for each step without fills, preventing it from learning to avoid trading entirely.

**Spread Capture vs Realized P&L**: We use LIFO-matched spread capture rather than average-cost realized P&L because:
- Directly measures round-trip profitability
- Dense signal: every completed round-trip provides feedback
- Rewards actual market making behavior (buy low, sell high)

---

## 🔬 Observation Space (36 signals)

### Market Microstructure (13)
Bid-ask spread, depth imbalance, order flow, volatility regime, price trend

### AMM Flow Signals (4)
Simulated AMM V3 concentrated liquidity signals: net flow, imbalance, inventory delta, cumulative flow

### Trade Feed Signals (8)
Real trade data: buy/sell volume, intensity, price impact, pressure indicators

### Agent State (11)
| Signal | Index | Purpose |
|--------|-------|---------|
| `leverage` | 25 | Current position risk |
| `position` | 26 | Normalized position size |
| `unrealized_pnl` | 27 | Mark-to-market P&L |
| `realized_pnl` | 28 | Locked-in P&L |
| `spread_capture` | 29 | LIFO round-trip profit |
| `deviation_from_target` | 30 | Distance from target inventory |
| `unrealized_pnl_pct` | 31 | P&L as % of position |
| `target_inventory_ema` | 32 | Smoothed target (what agent asked for) |
| `entry_price_distance` | 33 | Distance from avg entry to current mid |
| `time_since_last_fill` | 34 | Steps since last trade |
| `quote_mid_distance` | 35 | How far quotes are from mid (bps) |

---

## 🏗️ Architecture

### C++ Core (`litepool/rltrader/`)

| Component | Description |
|-----------|-------------|
| `sim_exchange` | Simulated exchange with realistic order matching |
| `position` | Position tracking with LIFO spread capture |
| `strategy` | Avellaneda-Stoikov quoting with RL-controlled parameters |
| `env_adaptor` | Bridges market data → RL observations |
| `amm_simulator` | Simulates AMM V3 for flow signals |
| `rltrader_litepool` | Gymnasium interface for Python |

### Python Agents (`litepool/rltrader/python/`)

| Component | Description |
|-----------|-------------|
| `inventory_agent.py` | MLP [64, 32] - learns target inventory (1 action) |
| `mm_agent.py` | MLP [128, 64] + LSTM - learns bid/ask spreads (2 actions) |
| `hierarchical_policy.py` | Coordinates both agents, combines 3 actions |
| `hierarchical_ppo.py` | Joint training with separate reward streams |
| `metric_logger.py` | TensorBoard logging for training analysis |

---

## ⚡ Quick Start

### Build

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cp lib/rltrader_litepool.so ../litepool/rltrader/
pip install -e ..
```

### Train

```bash
cd litepool/rltrader/python
python hierarchical_ppo.py
```

### Evaluate

```bash
python test_ppo.py   # Test data
python infer_ppo.py  # Production
```

### Monitor Training

```bash
tensorboard --logdir=runs/ --port=6006
# Open http://localhost:6006 in browser
```

---

## 📈 Training Output

```
================================================================================
Hierarchical PPO Training - Two-Agent Market Making
================================================================================
Inventory Agent: updates every 100 steps (50 sec)
MM Agent: updates every step (500ms)
Steps per epoch: 4096
Observations: 36 signals (13 market + 4 AMM + 8 trade + 11 agent state)
Actions: 3 (bid_spread, ask_spread, target_inventory)
Smart Requote: automatic when prices change >2 ticks
================================================================================

Epoch    10 | Step   40960 | MM.Rew  0.45 | Inv.Rew  2.31 | SprdCap $ 0.42 | ...

  [Episode] Env 0 | Steps  4096 | MM.Rew   1.65 | Inv.Rew   2.43 | SprdCap $ 0.18 | ...
```

| Metric | Description |
|--------|-------------|
| `MM.Rew` | Market Making reward (spread capture + fees) |
| `Inv.Rew` | Inventory reward (unrealized P&L delta) |
| `SprdCap` | LIFO spread capture (what MM optimizes) |
| `U.PnL` | Unrealized P&L (what Inventory optimizes) |

---

## 🎯 Action Space (3 dimensions)

| Action | Range | Description |
|--------|-------|-------------|
| `bid_spread` | [-1, 1] | Bid quote width (0.5x - 50x base spread) |
| `ask_spread` | [-1, 1] | Ask quote width (0.5x - 50x base spread) |
| `target_inventory` | [-1, 1] | Target leverage (±10%) |

Quote skew is automatically computed from `(current_leverage - target_inventory)` to push the position toward target.

### Smart Requote Logic

Instead of giving the agent a `requote` action (which it learned to game), requotes are triggered automatically:

| Condition | Behavior |
|-----------|----------|
| **Price change > 2 ticks** | Requote to track the market |
| **First step / No active orders** | Always requote |
| **After a fill** | Requote to replenish liquidity |

This prevents the agent from avoiding fills by never requoting, while reducing unnecessary order churn.

---

## 📁 Data Format

**Order Books** (`data/train_files/books/{n}.csv`):
```
timestamp,bid_price_1,bid_size_1,...,ask_price_1,ask_size_1,...
```

**Trades** (`data/train_files/trades/{n}.csv`):
```
exchange,symbol,timestamp,local_timestamp,id,side,price,amount
```

---

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_ENVS` | 6 | Parallel environments |
| `N_STEPS` | 4096 | Steps per epoch |
| `GAMMA` | 0.995 | Discount factor |
| `BASE_SPREAD_BPS` | 5.0 | Base spread (bps) |
| `ticks_per_step` | 5 | Ticks per RL step (500ms) |
| `inventory_update_freq` | 100 | Steps between inventory updates |
| `OBS_DIM` | 36 | Observation dimensions |
| `ACTION_DIM` | 3 | Action dimensions |
| `REQUOTE_TICK_THRESHOLD` | 2.0 | Ticks before auto-requote |
| `INACTIVITY_PENALTY` | 0.001 | MM penalty per step without fills |

---

## 📜 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built on the [EnvPool](https://github.com/sail-sg/envpool) architecture for high-performance vectorized environments.

Inspired by the [Avellaneda-Stoikov](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) market making framework.
