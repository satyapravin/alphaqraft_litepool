#pragma once
#include <deque>
#include "strategy.h"
#include "base_exchange.h"
#include "market_signal_builder.h"
#include "amm_simulator.h"

namespace RLTrader {

// Observation space dimension: 13 market signals + 4 AMM flow signals + 1 agent state
constexpr int OBS_DIM = 18;

class EnvAdaptor { 
public:
    EnvAdaptor(Strategy& strat, BaseExchange& exch, int ticks_per_step = 5);
    ~EnvAdaptor()  = default;
    
    void quote(const RLAction& action);

    void reset() ;
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
    std::array<double, OBS_DIM> state;  // 13 market + 3 AMM flow signals
    std::unordered_map<std::string, double> info;
    FixedVector<double, 20> bid_prices;
    FixedVector<double, 20> ask_prices;
    FixedVector<double, 20> bid_sizes;
    FixedVector<double, 20> ask_sizes;
};
}
