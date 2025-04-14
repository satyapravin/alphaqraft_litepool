#include "sim_exchange.h"
#include <vector>
#include <cassert>
#include <sstream>
#include <iostream>
#include "orderbook.h"

using namespace RLTrader;

std::vector<std::string> SimExchange::bid_price_labels(0);
std::vector<std::string> SimExchange::ask_price_labels(0);
std::vector<std::string> SimExchange::bid_size_labels(0);
std::vector<std::string> SimExchange::ask_size_labels(0);
bool SimExchange::init = SimExchange::initialize();

bool SimExchange::initialize() {
	for (int ii = 0; ii < 20; ++ii) {
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
			book.bid_sizes[ii]= lob.at(SimExchange::bid_size_labels[ii]);
			book.ask_sizes[ii] = lob.at(SimExchange::ask_size_labels[ii]);
		}
		ii++;
	}
}

SimExchange::SimExchange(const std::string& filename, long delay, int start_read, int max_read) :dataReader(filename, start_read, max_read), delay(delay) {
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
}

bool SimExchange::next_read(size_t& slot, OrderBook& book) {
    if (this->dataReader.hasNext()) {
        this->dataReader.next();
    	slot = 0;
    	toBook(this->dataReader.current().data, book);
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

void SimExchange::quote(std::string order_id, OrderSide side, const double& price, const double& amount) {
	Order order{};
	order.is_taker = false;
	order.microSecond = this->dataReader.getTimeStamp();
	order.amount = amount;
	order.orderId = order_id;
	order.price = price;
	order.side = side;
	order.state = OrderState::NEW;
	this->addToBuffer(order);
}

void SimExchange::market(std::string order_id, OrderSide side, const double &price, const double &amount, bool is_hedge) {
	Order order{};
	order.is_taker = true;
	order.microSecond = this->dataReader.getTimeStamp();
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
	for (auto &[fst, snd] : quotes) {
		if(snd.state != OrderState::FILLED
		   && snd.state != OrderState::CANCELLED
		   && snd.state != OrderState::CANCELLED_ACK) {
			snd.state = OrderState::CANCELLED;
			snd.microSecond = this->dataReader.getTimeStamp();
			this->addToBuffer(snd);
		   }
	}
}

void SimExchange::processPending(const DataRow& obs) {
	std::vector<long long> delete_stamps;
	std::vector<Order> bids;
	std::vector<Order> asks;


	for (auto& [timestamp, orders] : timed_buffer) {
		if (obs.id >= timestamp + delay) {
			for (Order& order : orders) {

				if (order.state == OrderState::NEW) {
					if (order.side == OrderSide::BUY) {
						if (order.price < obs.data.at("asks[0].price") || order.is_taker) {
							order.state = OrderState::NEW_ACK;
							bids.push_back(order);
						}
					}
					else {
						if (order.price > obs.data.at("bids[0].price") || order.is_taker) {
							order.state = OrderState::NEW_ACK;
							asks.push_back(order);
						}
					}
				}
				else if (order.state == OrderState::AMEND) {
					if (order.side == OrderSide::BUY) {
						auto& quotes = this->bid_quotes;
					
						if (quotes.find(order.orderId) != quotes.end()) {
							quotes[order.orderId].state = OrderState::AMEND_ACK;
							quotes[order.orderId] = order;
						}
					}
					else {
						auto& quotes = this->ask_quotes;
						if (quotes.find(order.orderId) != quotes.end()) {
							quotes[order.orderId].state = OrderState::AMEND_ACK;
							quotes[order.orderId] = order;
						}
					}
				}
				else if (order.state == OrderState::CANCELLED) {

					if (order.side == OrderSide::BUY) {
						auto it = this->bid_quotes.find(order.orderId);

						if (it != this->bid_quotes.end()) {
							it->second.state = OrderState::CANCELLED_ACK;
							this->bid_quotes.erase(order.orderId);
						}
					}
					else {
						auto it = this->ask_quotes.find(order.orderId);

						if (it != this->ask_quotes.end()) {
							it->second.state = OrderState::CANCELLED_ACK;
							this->ask_quotes.erase(order.orderId);
						}
					}
				}
				else {
					assert(order.state == OrderState::FILLED);
					executions.push_back(order);
				}
			}

			delete_stamps.push_back(timestamp);
		}
	}

	for (auto timestamp : delete_stamps)
		timed_buffer.erase(timestamp);

	for (const auto& order : bids) {
		this->bid_quotes[order.orderId] = order;
	}

	for(const auto& order: asks) {
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
	const DataRow& obs = this->dataReader.current();
	this->processPending(obs);

	std::vector<std::string> bids_filled;
	std::vector<std::string> asks_filled;

	for (auto& [order_id, order] : this->bid_quotes) {
		if (order.side == OrderSide::BUY && order.price > 0.00001 + this->dataReader.getDouble("bids[0].price") || order.is_taker) {
			order.state = OrderState::FILLED;
			if (order.is_taker) order.price = this->dataReader.getDouble("asks[0].price");
			bids_filled.push_back(order_id);
			this->addToBuffer(order);
		}
	}


	for(auto& [order_id, order] : this->ask_quotes) {
		if (order.side == OrderSide::SELL && order.price + 0.00001 < this->dataReader.getDouble("asks[0].price") || order.is_taker) {
			order.state = OrderState::FILLED;
			if (order.is_taker) order.price = this->dataReader.getDouble("bids[0].price");
			asks_filled.push_back(order_id);
			this->addToBuffer(order);
		}
	}


	for (auto& order_id : bids_filled) {
		this->bid_quotes.erase(order_id);
	}

	for (auto& order_id : asks_filled) {
		this->ask_quotes.erase(order_id);
	}
}
