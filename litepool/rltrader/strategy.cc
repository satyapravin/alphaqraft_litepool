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

void Strategy::quote(const std::vector<double>& buy_spreads,
                     const std::vector<double>& sell_spreads,
                     const std::vector<double>& buy_volumes,
                     const std::vector<double>& sell_volumes,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
	auto tick_size = instrument.getTickSize();
	auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
	auto leverage = posInfo.leverage;
        auto initBalance = position.getInitialBalance();
        auto skew = static_cast<int>(5 * leverage);
        auto bid_skew_multiplier = 1.0;
        auto ask_skew_multiplier  = 1.0;

        if (skew < 0) {
	    bid_skew_multiplier = 1.0 / std::abs(skew);
	    ask_skew_multiplier = std::abs(skew);
	} else if (skew > 0) {
	    bid_skew_multiplier = std::abs(skew);
	    ask_skew_multiplier = 1.0 / std::abs(skew);
	}

       	auto base_bid_price = bid_prices[0];
	auto base_ask_price = ask_prices[0];

	auto bid_width = 0.0001 * base_bid_price;
        auto ask_width = 0.0001 * base_ask_price;
        
	if (this->exchange.isDummy())	
            this->exchange.cancelOrders();

        for (auto ii = 0; ii < buy_spreads.size(); ++ii) {
             auto bid_spread = std::floor(buy_spreads[ii] * (ii+1) * bid_width / tick_size) * tick_size;
	     auto ask_spread = std::floor(sell_spreads[ii] * (ii+1) * ask_width / tick_size) * tick_size;
	     auto bid_quote = base_bid_price - bid_spread * bid_skew_multiplier;
	     auto ask_quote = base_ask_price + ask_spread * ask_skew_multiplier;
	     base_bid_price = std::min(base_bid_price, bid_quote);
	     base_ask_price = std::max(base_ask_price, ask_quote);
	     auto bid_size = (1 + buy_volumes[ii]) * initBalance * 0.01;
	     auto ask_size = (1 + sell_volumes[ii]) * initBalance * 0.01;

	     bid_size = instrument.getMinAmount() + instrument.getTradeAmount(bid_size, base_bid_price);
	     ask_size = instrument.getMinAmount() + instrument.getTradeAmount(ask_size, base_bid_price);
	     if (bid_size > 0 && base_bid_price <= bid_prices[0] + 1e-10) {
             	this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, base_bid_price, bid_size);
	     }
	     if (ask_size > 0 && base_ask_price >= ask_prices[0] - 1e-10) {
             	this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, base_ask_price, ask_size);
	     }
	}

}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
