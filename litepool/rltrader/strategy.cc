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

void Strategy::quote(const double& bid_spread,
                     const double& ask_spread,
                     const double& quote_range,
                     const double& target_q,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
	assert(bid_spread >= -1.0001 && bid_spread <= 1.001);
	assert(ask_spread >= -1.0001 && ask_spread <= 1.001);
	assert(quote_range >= -1.0001 && quote_range <= 1.001);
	assert(target_q >= -1.0001 && target_q <= 1.001);
	auto tick_size = instrument.getTickSize();
	auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
	auto leverage = posInfo.leverage;
        auto initBalance = position.getInitialBalance();
	auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
	auto q_target = (target_q - leverage);
	auto q_range = (quote_range + 1) / 5000;
        auto bid_price = mid_price * (1 - q_range * (bid_spread + 1.0));
        auto ask_price = mid_price * (1 + q_range * (ask_spread + 1.0));

        bid_price = std::floor(bid_price / tick_size) * tick_size;
	ask_price = std::ceil(ask_price / tick_size) * tick_size;

	if (bid_price > bid_prices[0]) bid_price = bid_prices[0];
	if (ask_price < ask_prices[0]) ask_price = ask_prices[0];

        auto bid_size = 0.05 * initBalance;
        auto ask_size = 0.05 * initBalance;

	if (q_target > 0) bid_size *= 1.5;
	if (q_target < 0) ask_size *= 1.5;
	
	bid_size = instrument.getTradeAmount(bid_size, bid_price);
	ask_size = instrument.getTradeAmount(ask_size, ask_price);
	if (exchange.isDummy()) exchange.cancelOrders();

        //std::cout << bid_price << "\t" << ask_price << "\t" << bid_size << "\t" << ask_size << std::endl;

	this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, bid_size);
        this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, ask_size);
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
