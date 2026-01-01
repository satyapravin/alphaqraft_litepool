// Copyright 2024 Alphaqraft
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <deque>
#include <memory>
#include "strategy.h"
#include "base_exchange.h"
#include "market_signal_builder.h"
#include "amm_simulator.h"
#include "trade_reader.h"
#include "trade_signal_builder.h"
#include "sim_exchange.h"

namespace RLTrader {

// Observation space: 13 market + 4 AMM flow + 8 trade + 11 agent state + 1 previous spread + 2 bid/ask distances 
//                   + 1 price_trend + 1 rolling_pnl + 1 pnl_volatility = 42 signals
constexpr int OBS_DIM = 42;

class EnvAdaptor { 
public:
    EnvAdaptor(Strategy& strat, BaseExchange& exch, const std::string& trade_filename = "", int ticks_per_step = 5);
    ~EnvAdaptor()  = default;
    
    void quote(const RLAction& action);

    void reset() ;
    void syncTradeReader(long long book_start_timestamp);  // Sync trade reader to book's starting timestamp
    bool next() ;  // Advances ticks_per_step ticks, processing fills each tick
    void getInfo(std::unordered_map<std::string, double>& info) ;
    void getState(std::array<double, OBS_DIM>& state) ;
private:
    void computeState(OrderBook& book);
    void computeInfo(OrderBook& book);
    Strategy& strategy;
    BaseExchange& exchange;
    int ticks_per_step_;  // Number of ticks to advance per RL step
    double max_unrealized_pnl = 0;
    double max_realized_pnl = 0;
    double drawdown = 0;
    double prev_mid_price_ = 0;
    std::deque<double> mid_price_deque;
    std::unique_ptr<MarketSignalBuilder> market_builder;
    AmmV3Simulator amm_simulator;  // AMM flow signal generator
    std::unique_ptr<TradeReader> trade_reader;  // Optional trade reader
    std::unique_ptr<TradeSignalBuilder> trade_signal_builder;  // Trade signal generator
    std::array<double, OBS_DIM> state;  // 13 market + 4 AMM flow + 8 trade + 11 agent state + 1 previous spread + 2 bid/ask distances + 1 mid_change
    
    // EMA smoothing for noisy AMM signals (inventory_delta and flow_imbalance)
    double flow_imbalance_ema_ = 0.0;
    double inventory_delta_ema_ = 0.0;
    static constexpr double AMM_SIGNAL_EMA_ALPHA = 0.1;  // ~7 step half-life for smoothing noisy signals
    
    // Rolling P&L momentum - EMA of per-step net P&L delta (using weighted avg unrealized)
    // Helps inventory agent see if it's on a winning or losing streak
    double prev_net_pnl_ = 0.0;           // Previous step's net P&L (realized + weighted avg unrealized)
    double rolling_pnl_momentum_ = 0.0;   // EMA of P&L deltas
    double rolling_pnl_var_ = 0.0;        // EMA of squared P&L deltas (for volatility)
    static constexpr double PNL_MOMENTUM_ALPHA = 0.05;  // ~14 step half-life (~7 sec window)
    
    // Price trend signal using dual EMAs (fast vs slow)
    // Positive = price trending up, Negative = trending down
    double price_ema_fast_ = 0.0;         // Fast EMA (~10 step half-life = 5 sec)
    double price_ema_slow_ = 0.0;         // Slow EMA (~60 step half-life = 30 sec)
    bool price_emas_initialized_ = false;
    static constexpr double PRICE_EMA_FAST_ALPHA = 0.07;   // ~10 step half-life
    static constexpr double PRICE_EMA_SLOW_ALPHA = 0.012;  // ~60 step half-life
    
    // Cached signals from last tick (updated in next() loop, used in computeState())
    std::vector<double> last_market_signals_;
    TradeSignals last_trade_signals_;
    std::unordered_map<std::string, double> info;
    FixedVector<double, 20> bid_prices;
    FixedVector<double, 20> ask_prices;
    FixedVector<double, 20> bid_sizes;
    FixedVector<double, 20> ask_sizes;
};
}
