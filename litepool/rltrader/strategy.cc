#include "strategy.h"
#include <string>
#include <cmath>
#include <cassert>
#include "orderbook.h"
#include "position_signal_builder.h"
#include <iostream>

using namespace RLTrader;

Strategy::Strategy(BaseInstrument& instr, BaseExchange& exch, const double& balance, int maxTicks)
	:instrument(instr), exchange(exch),
	 position(instr, balance, 0, 0),
	 order_id(0), max_ticks(maxTicks) {

	assert(max_ticks >= 5);
}

void Strategy::reset() {
	this->exchange.reset();
	double initQty = 0;
	double avgPrice = 0;
	this->exchange.fetchPosition(initQty, avgPrice);
        std::cout << "initial quantity=" << initQty << std::endl;
        std::cout << "initial price=" << avgPrice << std::endl;
	this->position.reset(initQty, avgPrice);
	this->order_id = 0;
}

void Strategy::quote(int buy_spread, int sell_spread, const double& buy_percent, const double& sell_percent,
                     FixedVector<double, 20>& bid_prices, FixedVector<double, 20>& ask_prices) {
	auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
	auto leverage = posInfo.leverage;
        auto initBalance = position.getInitialBalance();
        double buy_denom = 100;
        double sell_denom = 100;

	double buy_volume = initBalance * buy_percent / buy_denom;
	double sell_volume = initBalance * sell_percent / sell_denom;
        int skew = static_cast<int>(leverage);
        buy_spread = std::max(0, buy_spread + skew);
        sell_spread = std::max(0, sell_spread - skew);
        
        this->exchange.cancelOrders();

        if (buy_volume > 0 && buy_spread >= 0 && buy_spread < 20)
        	this->sendGrid(1, buy_spread, buy_volume, OrderSide::BUY, bid_prices);
        if (sell_volume > 0 && sell_spread >= 0 && sell_spread < 20)
        	this->sendGrid(1, sell_spread, sell_volume, OrderSide::SELL, ask_prices);
}

void Strategy::sendGrid(int levels, int start_level, const double& amount,
	                    OrderSide side, FixedVector<double, 20>& refPrices) {
	int spreads[4] = {0, 2, 4, 10};
        for (int ii = 0; ii < levels; ++ii) {
	     auto trade_amount = instrument.getTradeAmount(amount, refPrices[0]);
             if (trade_amount >= instrument.getMinAmount()) {
             	this->exchange.quote(std::to_string(++order_id), side, refPrices[ii + start_level + spreads[ii]], trade_amount);
             }
        }
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
