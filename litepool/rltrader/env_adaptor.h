#pragma once
#include "strategy.h"
#include "base_exchange.h"
#include "market_signal_builder.h"
#include "position_signal_builder.h"
#include "trade_signal_builder.h"

namespace RLTrader {
class EnvAdaptor { 
public:
    EnvAdaptor(Strategy& strat, BaseExchange& exch);
    ~EnvAdaptor()  = default;
    
    void quote(const std::vector<double>& buy_spreads, 
	       const std::vector<double>& sell_spreads, 
	       const std::vector<double>& buy_volumes, 
	       const std::vector<double>& sell_volumes);

    void reset() ;
    bool next() ;
    void getInfo(std::unordered_map<std::string, double>& info) ;
    void getState(std::array<double, 242*5>& state) ;
private:;
    void computeState(OrderBook& book);
    void computeInfo(OrderBook& book);
    Strategy& strategy;
    BaseExchange& exchange;
    double max_unrealized_pnl = 0;
    double max_realized_pnl = 0;
    double drawdown = 0;
    long num_trades = 0;
    std::unique_ptr<MarketSignalBuilder> market_builder;
    std::unique_ptr<PositionSignalBuilder> position_builder;
    std::unique_ptr<TradeSignalBuilder> trade_builder;
    std::array<double, 242*5> state;
    std::unordered_map<std::string, double> info;
    FixedVector<double, 20> bid_prices;
    FixedVector<double, 20> ask_prices;
    FixedVector<double, 20> bid_sizes;
    FixedVector<double, 20> ask_sizes;
};
}
