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
│  • Observes: AMM flow signals, trade pressure, P&L momentum, price trend  │
│  • Decides: Target inventory level (±100% leverage) + risk aversion (γ)   │
│  • Reward: LIFO unrealized P&L changes (learns market direction)          │
│  • Updates: Every 10 steps (5 seconds) - strategic time scale             │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ target_inventory, risk_aversion
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      MARKET MAKING AGENT (Tactical)                       │
│  "How do I efficiently reach that target while capturing spread?"         │
│                                                                           │
│  • Observes: Microstructure signals + target from Inventory Agent         │
│  • Decides: Bid/ask spreads (always requotes every step)                  │
│  • Reward: Spread capture + fee rebates                                   │
│  • Updates: Every step (1 second) - tactical time scale                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why This Works

| Benefit | Explanation |
|---------|-------------|
| **No Reward Gaming** | MM agent can't inflate rewards by only opening positions - it's paid for round-trips |
| **Clean Separation** | Inventory agent learns *when* to be long/short; MM agent learns *how* to get there |
| **Different Time Scales** | Strategic decisions (seconds) don't interfere with tactical execution (100ms ticks) |
| **Interpretable** | Can analyze each agent's learned behavior independently |
| **Trend + Spread** | Captures directional moves (via inventory) while earning spread (via MM) |
| **Risk Control** | Inventory agent also controls risk aversion (γ) for Avellaneda-Stoikov model |

---

## 📊 Reward Structure

The key insight: **each agent optimizes what it can control**.

| Agent | Reward | Controls | Signal |
|-------|--------|----------|--------|
| **Inventory** | `Δ(lifo_unrealized_pnl)` | When to be long/short/flat, risk level | Market direction |
| **MM** | `spread_capture + fees` | Quote pricing | Execution quality |

```cpp
// MM Agent: optimizes spread capture (LIFO) + fee rebates
mm_reward = (spread_capture_delta + fee_delta) × REWARD_SCALE

// Inventory Agent: optimizes LIFO unrealized P&L (market direction)
inv_reward = lifo_unrealized_pnl_delta × REWARD_SCALE
```

**LIFO Accounting**: Both agents use Last-In-First-Out matching for P&L:
- Directly measures round-trip profitability
- Dense signal: every completed round-trip provides feedback
- Rewards actual market making behavior (buy low, sell high)

**Fee Rebates**: Negative fees mean the agent EARNS rebates (maker orders). The `maker_fee = -0.000025` means earning 2.5 bps on each fill.

---

## 🔬 Observation Space (42 signals)

### Market Microstructure (13)
| Index | Signal | Description |
|-------|--------|-------------|
| 0 | `bid_ask_spread` | Current spread in bps |
| 1 | `depth_imbalance` | (bid_depth - ask_depth) / total |
| 2-5 | `order_flow_*` | Directional flow indicators |
| 6-8 | `volatility_*` | Short/medium/long volatility |
| 9-12 | `trend_*` | Price trend indicators |

### AMM Flow Signals (4)
| Index | Signal | Description |
|-------|--------|-------------|
| 13 | `amm_net_flow` | Net AMM trading flow |
| 14 | `amm_imbalance` | AMM inventory imbalance |
| 15 | `amm_inventory_delta` | Change in AMM inventory |
| 16 | `amm_cumulative_flow` | Decaying cumulative flow (60s half-life) |

### Trade Feed Signals (8)
| Index | Signal | Description |
|-------|--------|-------------|
| 17-18 | `buy/sell_volume` | Trade volumes by side |
| 19-20 | `buy/sell_intensity` | Trade frequency |
| 21-22 | `price_impact_*` | Market impact estimates |
| 23-24 | `pressure_*` | Buying/selling pressure |

### Agent State (17)
| Index | Signal | Description |
|-------|--------|-------------|
| 25 | `leverage` | Current position leverage |
| 26 | `position` | Normalized position size |
| 27 | `unrealized_pnl` | Mark-to-market P&L (weighted avg) |
| 28 | `realized_pnl` | Locked-in P&L (weighted avg) |
| 29 | `spread_capture` | LIFO round-trip profit |
| 30 | `deviation_from_target` | Distance from target inventory |
| 31 | `unrealized_pnl_pct` | P&L as % of position |
| 32 | `target_inventory_ema` | Smoothed target (what agent asked for) |
| 33 | `entry_price_distance` | Distance from avg entry to current mid |
| 34 | `time_since_last_fill` | Steps since last trade |
| 35 | `quote_mid_distance` | How far quotes are from mid (bps) |
| 36 | `lifo_unrealized_pnl` | LIFO unrealized P&L |
| 37 | `lifo_realized_pnl` | LIFO realized P&L |
| 38 | `fees` | Cumulative fees (negative = rebates earned) |
| 39 | `price_trend` | Dual EMA price trend signal |
| 40 | `rolling_pnl_momentum` | EMA of P&L changes |
| 41 | `rolling_pnl_volatility` | Volatility of P&L changes |

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
| `inventory_agent.py` | Shared Encoder + LSTM - learns target inventory + risk aversion (2 actions) |
| `mm_agent.py` | Shared Encoder + Attention + LSTM - learns bid/ask spreads (2 actions) |
| `hierarchical_policy.py` | Coordinates both agents, combines 4 actions |
| `hierarchical_ppo.py` | Joint PPO training with separate reward streams |
| `shared_encoder.py` | MLP encoder shared between agents |

---

## ⚡ Quick Start

### Build

```bash
pip install -e . --no-build-isolation
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
Inventory Agent: updates every 10 steps (5 sec)
MM Agent: updates every step (1 sec)
Steps per epoch: 4096
Observations: 42 signals
Actions: 4 (bid_spread, ask_spread, target_inventory, risk_aversion)
Requote: always requote every step (no agent control)
================================================================================

Epoch    10 | Step   40960 | MM.Rew  0.45 | Inv.Rew  2.31 | SprdCap $ 0.42 | ...

  [Episode] Env 0 | Steps  4096 | MM.Rew   1.65 | Inv.Rew   2.43 | SprdCap $ 0.18 | ...
```

| Metric | Description |
|--------|-------------|
| `MM.Rew` | Market Making reward (spread capture + fees) |
| `Inv.Rew` | Inventory reward (LIFO unrealized P&L delta) |
| `SprdCap` | LIFO spread capture (what MM optimizes) |
| `U.PnL` | Unrealized P&L (what Inventory optimizes) |

---

## 🎯 Action Space (4 dimensions)

| Action | Range | Agent | Description |
|--------|-------|-------|-------------|
| `bid_spread` | [0, 1] | MM | Bid quote width multiplier |
| `ask_spread` | [0, 1] | MM | Ask quote width multiplier |
| `target_inventory` | [-1, 1] | Inventory | Target leverage (±100%) |
| `risk_aversion` | [0, 0.1] | Inventory | Avellaneda-Stoikov γ parameter |

### Quoting Mechanism

The strategy uses Avellaneda-Stoikov with inventory skew:
- Base spread determined by `base_spread_bps` config
- Agent adjusts spread via `bid_spread` and `ask_spread` multipliers
- Inventory skew automatically pushes quotes to reduce position toward target
- Emergency skew activates when leverage exceeds threshold (1x) to aggressively offload

### Ladder Quoting

Orders are placed at multiple price levels (10 levels per side):
- Level 1: 1x base spread (closest to mid)
- Level 2: 2x base spread
- ...
- Level 10: 10x base spread (furthest from mid)

This provides:
- Natural dollar-cost averaging on large moves
- Reduced adverse selection (only closest levels fill on small moves)
- More rebate opportunities (multiple fills)

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
| `num_envs` | 8 | Parallel environments |
| `n_steps` | 4096 | Steps per epoch |
| `gamma` | 0.99 | Discount factor |
| `base_spread_bps` | 2.0 | Base spread (bps) |
| `ticks_per_step` | 10 | Ticks per RL step (1 second) |
| `inventory_update_freq` | 10 | Steps between inventory updates |
| `OBS_DIM` | 42 | Observation dimensions |
| `ACTION_DIM` | 4 | Action dimensions |
| `max_leverage` | 3.0 | Maximum allowed leverage |
| `initial_balance` | 10000 | Starting balance (USD) |

---

## 📜 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built on the [EnvPool](https://github.com/sail-sg/envpool) architecture for high-performance vectorized environments.

Inspired by the [Avellaneda-Stoikov](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) market making framework.
