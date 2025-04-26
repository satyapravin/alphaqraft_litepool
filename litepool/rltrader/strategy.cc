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
        std::cout << "initial quantity=" << initQty << std::endl;
        std::cout << "initial price=" << avgPrice << std::endl;
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
	assert(bid_size >= -1.0001 && bid_size <= 1.001);
	assert(ask_size >= -1.0001 && ask_size <= 1.001);
	auto tick_size = instrument.getTickSize();
	auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
	auto leverage = posInfo.leverage;
        auto initBalance = position.getInitialBalance();
        	
	auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
	
        auto bid_price = bid_prices[0] - 25 * (bid_spread + 1.0) * tick_size;
        auto ask_price = ask_prices[0] + 25 * (ask_spread + 1.0) * tick_size;

	auto bid_size_0 = 0.005 * initBalance + (bid_size + 1.0) * 0.02 * initBalance;
	auto ask_size_0 = 0.005 * initBalance + (ask_size + 1.0) * 0.02 * initBalance;

	/*
	double spot_position = 0;
	double spot_avg_price = 0;

        this->exchange.fetchPosition(spot_position, spot_avg_price, true);
        double spotLeverage = spot_position / initBalance / mid_price;

	if (std::isnan(spotLeverage) || std::isinf(spotLeverage)) return; // safety check
									    
	double diffLeverage = spotLeverage + leverage / 2;

        if (!exchange.isDummy()) {	
	    std::cout << "Hedge position: " <<  spot_position 
	              << "\t Hedge leverage: " << spotLeverage 
		      << "\t Diff leverage: " << diffLeverage << std::endl;
        }

	if (!exchange.isDummy() && std::abs(diffLeverage) > 0.05) { 
	    double new_spot = 0;
	    this->exchange.cancelOrders();
	    std::this_thread::sleep_for(std::chrono::milliseconds(100));
	    this->exchange.fetchPosition(new_spot, spot_avg_price, true);

	    if (std::abs(new_spot - spot_position) > 0.05) return;

	    auto spotSize = std::abs(diffLeverage) * initBalance;
            auto spotSide = diffLeverage > 0 ? OrderSide::SELL : OrderSide::BUY;
	    auto hedge_price = std::floor(mid_price / 0.25) * 0.25;
	    auto amount = instrument.getTradeAmount(spotSize, hedge_price);
	    std::cout << "New hedging leverage= " << std::abs(diffLeverage) << "\t side= "  << spotSide 
		      << "\t price= " << hedge_price << "\t size= " << amount << std::endl;
	    this->exchange.market(std::to_string(++order_id), spotSide, hedge_price, amount, true); 
	}
	*/

	auto skew = leverage * 20 * tick_size;

	bid_price -= skew;
	ask_price -= skew;

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
		//std::cout << "Bids" << "\t" << bid_price << "\t" << bid_size << std::endl;
	        this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, bid_size);
	    }
	    else {
		std::cout << "BIDS ARE ZERO\t" << bid_price << "\t" << bid_size_0 << std::endl;
	    }
	    if (ask_size > 0) {
		//std::cout << "Asks" << "\t" << ask_price << "\t" << ask_size << std::endl;
                this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, ask_size);
	    }
	    else {
		std::cout << "ASKS ARE ZERO\t" << ask_price << "\t" << ask_size_0 << std::endl;
	    }
	}
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
