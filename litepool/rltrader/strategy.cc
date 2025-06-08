#include "strategy.h"
#include <cassert>
#include <string>
#include <cmath>
#include <cassert>
#include "orderbook.h"
#include "position_signal_builder.h"
#include <iostream>
#include <thread>
#include <chrono>

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
	this->exchange.fetchPosition(initQty, avgPrice, false);

	if (!exchange.isDummy()) {
            std::cout << "initial quantity=" << initQty << std::endl;
            std::cout << "initial price=" << avgPrice << std::endl;
	}

	this->position.reset(initQty, avgPrice);
	this->order_id = 0;
}

void Strategy::quote(const double& bid_spread,
		     const double& ask_spread,
		     const double& bid_size,
                     const double& ask_size,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
	assert(bid_spread >= -1.0001 && bid_spread <= 1.001);
	assert(ask_spread >= -1.0001 && ask_spread <= 1.001);
	assert(ask_size >= -1.0001 && ask_size <= 1.001);
	assert(ask_size >= -1.0001 && ask_size <= 1.001);
	auto tick_size = instrument.getTickSize();
	auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
	auto leverage = posInfo.leverage;
        auto initBalance = position.getInitialBalance();
        	
	auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
	
        auto bid_price = bid_prices[0] - (1.0 + bid_spread) *  50 * tick_size;
        auto ask_price = ask_prices[0] + (1.0 + ask_spread) *  50 * tick_size;

	auto bid_size_0 = (1.01 + bid_size) / 20.0 * initBalance;
	auto ask_size_0 = (1.01 + ask_size) / 20.0 * initBalance;

	auto bid_price_0 = std::floor(bid_price / tick_size) * tick_size;
	auto ask_price_0 = std::ceil(ask_price / tick_size) * tick_size;

	if (bid_price_0 > bid_prices[0]) bid_price_0 = bid_prices[0];
	if (ask_price_0 < ask_prices[0]) ask_price_0 = ask_prices[0];

        exchange.cancelOrders();

	for (int ii=0; ii < 1; ++ii) {
	    bid_price = bid_price_0 - 8 * tick_size * ii;
	    ask_price = ask_price_0 + 8 * tick_size * ii;
	    auto bid_size = instrument.getTradeAmount(bid_size_0 * (1 + 0.2 * ii), bid_price);
	    auto ask_size = instrument.getTradeAmount(ask_size_0 * (1 + 0.2 * ii), ask_price);
	    if (bid_size > 0) {
	        this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, bid_size);
	    }
	    else {
		//std::cout << "BIDS ARE ZERO\t" << bid_price << "\t" << bid_size_0 << std::endl;
	    }
	    if (ask_size > 0) {
                this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, ask_size);
	    }
	    else {
		//std::cout << "ASKS ARE ZERO\t" << ask_price << "\t" << ask_size_0 << std::endl;
	    }
	}
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
