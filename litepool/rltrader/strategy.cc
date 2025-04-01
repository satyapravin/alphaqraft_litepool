#include "strategy.h"
#include <cassert>
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

void Strategy::quote(const double& mid_spread,
                     const double& skew_multiplier,
                     const double& target_q,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
	auto tick_size = instrument.getTickSize();
	auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
	auto leverage = posInfo.leverage;
        auto initBalance = position.getInitialBalance();
	auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
	auto q_target = (target_q - leverage) * initBalance;
	auto spread_opt = mid_spread * 20 * tick_size / 2.0;
	auto inventory_skew = skew_multiplier * 10 * q_target * tick_size;
        auto bid_price = mid_price - spread_opt + inventory_skew;
        auto ask_price = mid_price + spread_opt + inventory_skew;

        bid_price = std::floor(bid_price / tick_size) * tick_size;
	ask_price = std::ceil(ask_price / tick_size) * tick_size;

	if (bid_price > bid_prices[0]) bid_price = bid_prices[0];
	if (ask_price < ask_prices[0]) ask_price = ask_prices[0];

	auto bid_size = instrument.getTradeAmount(0.02 * initBalance, bid_price);
	auto ask_size = instrument.getTradeAmount(0.02 * initBalance, ask_price);

	if (q_target > 0) bid_size *= 2;
	if (q_target < 0) ask_size *= 2;
	if (exchange.isDummy()) exchange.cancelOrders();
	    
	this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, bid_size);
        this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, ask_size);
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
