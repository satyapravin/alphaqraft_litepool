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

#include "sim_exchange.h"
#include <vector>
#include <cassert>
#include <sstream>
#include <iostream>
#include "orderbook.h"

namespace RLTrader {

namespace {
    constexpr size_t MAX_BOOK_LEVELS = 20;
}

std::vector<std::string> SimExchange::bid_price_labels(0);
std::vector<std::string> SimExchange::ask_price_labels(0);
std::vector<std::string> SimExchange::bid_size_labels(0);
std::vector<std::string> SimExchange::ask_size_labels(0);
bool SimExchange::init = SimExchange::initialize();

bool SimExchange::initialize() {
	for (size_t ii = 0; ii < MAX_BOOK_LEVELS; ++ii) {
		std::ostringstream bid_price_lbl;
		bid_price_lbl << "bids[" << ii << "].price";
		std::ostringstream ask_price_lbl;
		ask_price_lbl << "asks[" << ii << "].price";
		std::ostringstream bid_amount_lbl;
		bid_amount_lbl << "bids[" << ii << "].amount";
		std::ostringstream ask_amount_lbl;
		ask_amount_lbl << "asks[" << ii << "].amount";

		SimExchange::bid_price_labels.push_back(bid_price_lbl.str());
		SimExchange::ask_price_labels.push_back(ask_price_lbl.str());
		SimExchange::bid_size_labels.push_back(bid_amount_lbl.str());
		SimExchange::ask_size_labels.push_back(ask_amount_lbl.str());
	}

	return true;
}

void SimExchange::toBook(const std::unordered_map<std::string, double>& lob, OrderBook& book)  {
	int ii = 0;
	for(const auto & bid_price_label : bid_price_labels) {
		if (lob.find(bid_price_label) != lob.end()) {
			book.bid_prices[ii] = lob.at(SimExchange::bid_price_labels[ii]);
			book.ask_prices[ii] = lob.at(SimExchange::ask_price_labels[ii]);
			book.bid_sizes[ii] = lob.at(SimExchange::bid_size_labels[ii]);
			book.ask_sizes[ii] = lob.at(SimExchange::ask_size_labels[ii]);
		} else {
			book.bid_prices[ii] = 0.0;
			book.ask_prices[ii] = 0.0;
			book.bid_sizes[ii] = 0.0;
			book.ask_sizes[ii] = 0.0;
		}
		ii++;
	}
}

SimExchange::SimExchange(const std::string& filename, long delay, int start_read) :dataReader(filename, start_read), delay(delay), current_timestamp(0) {
	bid_quotes.clear();
	ask_quotes.clear();
	executions.clear();
	timed_buffer.clear();
	dataReader.reset();
}


void SimExchange::reset() {
	this->dataReader.reset();
	this->executions.clear();
	this->bid_quotes.clear();
	this->ask_quotes.clear();
	this->timed_buffer.clear();
	current_timestamp = 0;  // Reset cached timestamp
}

bool SimExchange::next_read(size_t& slot, OrderBook& book) {
    if (this->dataReader.hasNext()) {
        this->dataReader.next();
    	slot = 0;
    	const DataRow& current_row = this->dataReader.current();
    	// Cache timestamp when iterator is valid
    	current_timestamp = current_row.id;
    	toBook(current_row.data, book);
        this->execute();
    } else {
        return false;
    }

    return true;
}

void SimExchange::done_read(size_t slot) {
	// no ops
}

const std::map<std::string, Order>& SimExchange::getBidOrders() const {
	return this->bid_quotes;
}

const std::map<std::string, Order>& SimExchange::getAskOrders() const {
	return this->ask_quotes;
}

std::vector<Order> SimExchange::getUnackedOrders() const {
	std::vector<Order> retval;
	
	for (auto& [timestamp, orders] : this->timed_buffer) {
		retval.insert(retval.end(), orders.begin(), orders.end());
	}

	return retval;
}

void SimExchange::fetchPosition(double &posAmount, double &avgPrice, bool is_hedge) {
	posAmount = 0;
	avgPrice = 0;
}

void SimExchange::quote(const std::string& order_id, OrderSide side, double price, double amount) {
	Order order{};
	order.is_taker = false;
	order.microSecond = current_timestamp;  // Use cached timestamp instead of getTimeStamp()
	order.amount = amount;
	order.orderId = order_id;
	order.price = price;
	order.side = side;
	order.state = OrderState::NEW;
	this->addToBuffer(order);
}

void SimExchange::market(const std::string& order_id, OrderSide side, double price, double amount, bool /*is_hedge*/) {
	Order order{};
	order.is_taker = true;
	order.microSecond = current_timestamp;  // Use cached timestamp instead of getTimeStamp()
	order.amount = amount;
	order.orderId = order_id;
	order.price = price;
	order.side = side;
	order.state = OrderState::NEW;
        this->addToBuffer(order);
}


std::vector<Order> SimExchange::getFills() {
	std::vector<Order> retval(this->executions);
	this->executions.clear();
	return retval;
}

void SimExchange::cancel(std::map<std::string, Order>& quotes) {
	// CRITICAL FIX: Don't mark orders as CANCELLED in quotes map immediately
	// This prevents execute() from skipping orders that can fill
	// Instead, create a cancellation order and add it to timed_buffer
	// The cancellation will be processed after the delay period, respecting latency
	for (auto &[fst, snd] : quotes) {
		if(snd.state != OrderState::FILLED
		   && snd.state != OrderState::CANCELLED
		   && snd.state != OrderState::CANCELLED_ACK) {
			// Create a cancellation order (don't modify the original order in quotes)
			// This allows the order to still fill in execute() until cancellation is processed
			Order cancel_order = snd;  // Copy the order
			cancel_order.state = OrderState::CANCELLED;
			cancel_order.microSecond = current_timestamp;  // Use cached timestamp
			this->addToBuffer(cancel_order);
		   }
	}
}

void SimExchange::processPending(const DataRow& obs) {
	std::vector<long long> delete_stamps;
	std::vector<Order> bids_to_add;
	std::vector<Order> asks_to_add;
	
	// Safety check: ensure price data exists before accessing
	if (obs.data.count("asks[0].price") == 0 || obs.data.count("bids[0].price") == 0) {
		return;  // Skip processing if market data is missing
	}
	const double best_ask = obs.data.at("asks[0].price");
	const double best_bid = obs.data.at("bids[0].price");

	for (auto& [timestamp, orders] : timed_buffer) {
		if (obs.id >= timestamp + delay) {
			for (Order& order : orders) {

				if (order.state == OrderState::NEW) {
					if (order.side == OrderSide::BUY) {
						if (order.is_taker) {
							// Market order: accept and fill immediately in execute()
							order.state = OrderState::NEW_ACK;
							bids_to_add.push_back(order);
						}
						else if (order.price < best_ask) {
							// Passive limit order: price below ask, accepted
							order.state = OrderState::NEW_ACK;
							bids_to_add.push_back(order);
						}
						// else: POST_ONLY order would cross (price >= ask), reject silently
					}
					else {
						if (order.is_taker) {
							// Market order: accept and fill immediately in execute()
							order.state = OrderState::NEW_ACK;
							asks_to_add.push_back(order);
						}
						else if (order.price > best_bid) {
							// Passive limit order: price above bid, accepted
							order.state = OrderState::NEW_ACK;
							asks_to_add.push_back(order);
						}
						// else: POST_ONLY order would cross (price <= bid), reject silently
					}
				}
				else if (order.state == OrderState::CANCELLED) {
					auto& quotes = (order.side == OrderSide::BUY) ? this->bid_quotes : this->ask_quotes;
					auto it = quotes.find(order.orderId);
					if (it != quotes.end()) {
						// Process cancellation: remove order from quotes
						// Note: We respect the cancellation request even if market has crossed
						// The cancellation was requested earlier (with latency), so it takes precedence
						// If the order could have filled, it should have filled in execute() before
						// the cancellation was processed (after its delay period)
						it->second.state = OrderState::CANCELLED_ACK;
						quotes.erase(it);
					}
				}
				else if (order.state == OrderState::FILLED) {
					executions.push_back(order);
				}
			}

			delete_stamps.push_back(timestamp);
		}
	}

	for (auto timestamp : delete_stamps) {
		timed_buffer.erase(timestamp);
	}

	for (const auto& order : bids_to_add) {
		this->bid_quotes[order.orderId] = order;
	}

	for (const auto& order : asks_to_add) {
		this->ask_quotes[order.orderId] = order;
	}
}

void SimExchange::cancelOrders() {
	this->cancel(this->bid_quotes);
	this->cancel(this->ask_quotes);
}

void SimExchange::addToBuffer(const Order& order) {
	if (this->timed_buffer.find(order.microSecond) == this->timed_buffer.end()) {
		this->timed_buffer[order.microSecond] = std::vector<Order>();
	}

	this->timed_buffer[order.microSecond].push_back(order);
}

void SimExchange::execute() {
	// Get current observation - this should be valid since next_read() just called next()
	const DataRow& obs = this->dataReader.current();
	this->processPending(obs);

	std::vector<std::string> bids_filled;
	std::vector<std::string> asks_filled;
	
	// Get best bid/ask from current observation
	// IMPORTANT: Check for valid prices (> 0) to avoid filling on missing data
	const double best_ask = obs.data.count("asks[0].price") > 0 ? obs.data.at("asks[0].price") : 0.0;
	const double best_bid = obs.data.count("bids[0].price") > 0 ? obs.data.at("bids[0].price") : 0.0;
	
	// Skip fill checking if market data is invalid
	if (best_ask <= 0 || best_bid <= 0) {
		return;
	}

	// BUY orders fill when ask crosses down to our price
	for (auto& [order_id, order] : this->bid_quotes) {
		// Skip orders that are already cancelled (pending removal)
		if (order.state == OrderState::CANCELLED || order.state == OrderState::CANCELLED_ACK) {
			continue;
		}
		
		// Taker: fill immediately at best ask
		// Maker: fill when best_ask <= our_bid (someone willing to sell at our price)
		if (order.is_taker || best_ask <= order.price) {
			order.state = OrderState::FILLED;
			order.price = order.is_taker ? best_ask : order.price;  // Taker at ask, maker at limit
			bids_filled.push_back(order_id);
			this->addToBuffer(order);
		}
	}

	// SELL orders fill when bid crosses up to our price
	for (auto& [order_id, order] : this->ask_quotes) {
		// Skip orders that are already cancelled (pending removal)
		if (order.state == OrderState::CANCELLED || order.state == OrderState::CANCELLED_ACK) {
			continue;
		}
		
		bool should_fill = order.is_taker || best_bid >= order.price;
		if (should_fill && order.state != OrderState::FILLED) {
			// Order should fill - mark it as filled
			order.state = OrderState::FILLED;
			order.price = order.is_taker ? best_bid : order.price;  // Taker at bid, maker at limit
			asks_filled.push_back(order_id);
			this->addToBuffer(order);
		}
	}

	for (const auto& order_id : bids_filled) {
		this->bid_quotes.erase(order_id);
	}

	for (const auto& order_id : asks_filled) {
		this->ask_quotes.erase(order_id);
	}
}

} // namespace RLTrader
