#include "deribit_exchange.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cctype>

using namespace RLTrader;

bool caseInsensitiveCompare(const std::string& a, const std::string& b) {
    return a.size() == b.size() && 
           std::equal(a.begin(), a.end(), b.begin(),
               [](char a, char b) {
                   return std::tolower(a) == std::tolower(b);
               });
}


DeribitExchange::DeribitExchange(const std::string& symbol, const std::string& hedge_symbol, const std::string& api_key, const std::string& api_secret)
    :db_client(api_key, api_secret, symbol, hedge_symbol), symbol(symbol), hedge_symbol(hedge_symbol), orders_count(0), RESTApi(api_key, api_secret)
{
}

void DeribitExchange::toBook(const std::unordered_map<std::string, double>& lob, OrderBook& book)  {
    throw std::runtime_error("cannot construct OrderBook from unordered_map in prod exchange");
}

void DeribitExchange::reset() {
    db_client.stop();
    std::lock_guard<std::mutex> lock(this->fill_mutex);
    this->executions.clear();
    this->set_callbacks();
    db_client.start();
}

void DeribitExchange::set_callbacks() {
    this->db_client.set_private_trade_cb([this] (const json& data) { handle_private_trade_updates(data); });
    this->db_client.set_OrderBook_cb([this] (const json& data) { handle_order_book_updates(data); });
    this->db_client.set_order_cb([this] (const json& data) { handle_order_updates(data);});
    this->db_client.set_position_cb([this] (const json& data) { handle_position_updates(data);});
}

void DeribitExchange::handle_private_trade_updates (const json& json_array) {
    for (auto ii=0; ii < json_array.size(); ++ii) {
        const auto& data = json_array[ii];
        if (data["instrument_name"] == this->symbol) {
            Order order;
            order.amount = data["amount"];
            order.is_taker = !data["post_only"];
            order.price = data["price"];
            order.side = caseInsensitiveCompare(data["direction"], "buy")  ? OrderSide::BUY : OrderSide::SELL;
            order.state = OrderState::FILLED;
            order.orderId = data["order_id"];
            order.microSecond = data["timestamp"];
            std::string trade_id = data["trade_id"];
            std::cout << data["amount"] << "\t" << data["direction"] << "\t" << data["trade_id"] << std::endl;
            std::lock_guard<std::mutex> lock(this->fill_mutex);
            if (processed_trades.find(trade_id) == processed_trades.end()) {
    	        processed_trades.insert(trade_id);
                this->executions.push_back(order);
            }
        } else {
	    std::cout << "invalid instrument: " << data["instrument_name"] << std::endl;
	}
    }
}

void DeribitExchange::handle_order_book_updates (const json& data) {
    size_t write_slot;
    auto& book = this->book_buffer.get_write_slot(write_slot);
    size_t idx = 0;
    for (const auto& bid : data["bids"]) {
        book.bid_prices[idx] = bid[0].get<double>();
        book.bid_sizes[idx] = bid[1].get<double>();
        ++idx;
	if (idx >= 20) break;
    }

    idx = 0;
    for (const auto& ask : data["asks"]) {
        book.ask_prices[idx] = ask[0].get<double>();
        book.ask_sizes[idx] = ask[1].get<double>();
	++idx;
	if (idx >= 20) break;
    }

    this->book_buffer.commit_write(write_slot);
}


void DeribitExchange::handle_order_updates (const json& data) {
    ++this->orders_count;
    if (data["instrument_name"] == symbol) {
        OrderSide side = data["direction"] == "buy" ? OrderSide::BUY: OrderSide::SELL;
        double price = data["price"];
        const std::string& order_id = data["order_id"];
        OrderState order_state = (data["order_state"] == "open"
                                 || data["order_state"] == "untriggered") ? OrderState::NEW_ACK : OrderState::CANCELLED;
    }
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void DeribitExchange::handle_position_updates (const json& data) { // NOLINT(*-convert-member-functions-to-static)
    //std::cout << data << std::endl;
}

bool DeribitExchange::next_read(size_t& slot, OrderBook& book) {
    book = this->book_buffer.get_read_slot(slot);
    return true;
}

void DeribitExchange::fetchPosition(double &posAmount, double &avgPrice, bool is_hedge) {
    if (is_hedge) 
        RESTApi.fetch_position(hedge_symbol, posAmount, avgPrice);
    else
        RESTApi.fetch_position(symbol, posAmount, avgPrice);
}

void DeribitExchange::cancelOrders() {
    db_client.cancel_all_orders();
}


std::vector<Order> DeribitExchange::getFills() {
    std::lock_guard<std::mutex> lock(this->fill_mutex);
    std::vector<Order> fills(std::move(this->executions));
    this->executions.clear();
    return fills;
}

bool is_close(const double& a, const double& b) {
    return std::abs(a - b) < 0.001;
}

void DeribitExchange::quote(std::string order_id, OrderSide side, const double& price, const double& amount) {
    std::string sidestr = side == OrderSide::BUY ? "buy" : "sell";
    //this->db_client.cancel_all_by_label(sidestr);
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    this->db_client.place_order(sidestr, price, amount, sidestr, false, "limit");
}

void DeribitExchange::market(std::string order_id, OrderSide side, const double& price, const double& amount, bool is_hedge) {
    std::string sidestr = side == OrderSide::BUY ? "buy" : "sell";
    this->db_client.place_order(sidestr, price, amount, sidestr, is_hedge, "market");

}
