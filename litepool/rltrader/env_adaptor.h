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

// Observation space: 13 market + 4 AMM flow + 8 trade + 7 agent state = 32 signals
constexpr int OBS_DIM = 32;

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
    std::deque<double> mid_price_deque;
    std::unique_ptr<MarketSignalBuilder> market_builder;
    AmmV3Simulator amm_simulator;  // AMM flow signal generator
    std::unique_ptr<TradeReader> trade_reader;  // Optional trade reader
    std::unique_ptr<TradeSignalBuilder> trade_signal_builder;  // Trade signal generator
    std::array<double, OBS_DIM> state;  // 13 market + 4 AMM flow + 8 trade + 7 agent state
    std::unordered_map<std::string, double> info;
    FixedVector<double, 20> bid_prices;
    FixedVector<double, 20> ask_prices;
    FixedVector<double, 20> bid_sizes;
    FixedVector<double, 20> ask_sizes;
};
}
