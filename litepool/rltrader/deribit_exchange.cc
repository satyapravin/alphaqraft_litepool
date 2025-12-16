#include "deribit_exchange.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <cctype>

namespace RLTrader {

namespace {
    // Constants
    constexpr size_t MAX_BOOK_LEVELS = 20;
    constexpr auto RESET_DELAY = std::chrono::seconds(5);
    
    bool caseInsensitiveCompare(const std::string& a, const std::string& b) {
        return a.size() == b.size() && 
               std::equal(a.begin(), a.end(), b.begin(),
                   [](char ca, char cb) {
                       return std::tolower(ca) == std::tolower(cb);
                   });
    }
}

DeribitExchange::DeribitExchange(const std::string& symbol, const std::string& hedge_symbol, 
                                 const std::string& api_key, const std::string& api_secret)
    : db_client(api_key, api_secret, symbol, hedge_symbol), 
      RESTApi(api_key, api_secret),
      symbol(symbol), 
      hedge_symbol(hedge_symbol)
{
}

void DeribitExchange::toBook(const std::unordered_map<std::string, double>& /*lob*/, OrderBook& /*book*/) {
    throw std::runtime_error("Cannot construct OrderBook from unordered_map in prod exchange");
}

void DeribitExchange::reset() {
    db_client.stop();
    std::this_thread::sleep_for(RESET_DELAY);
    {
        std::lock_guard<std::mutex> lock(this->fill_mutex);
        this->executions.clear();
        this->processed_trades.clear();
    }
    this->set_callbacks();
    db_client.start();
    std::this_thread::sleep_for(RESET_DELAY);
}

void DeribitExchange::set_callbacks() {
    this->db_client.set_private_trade_cb([this](const json& data) { handle_private_trade_updates(data); });
    this->db_client.set_OrderBook_cb([this](const json& data) { handle_order_book_updates(data); });
    this->db_client.set_order_cb([this](const json& data) { handle_order_updates(data); });
    this->db_client.set_position_cb([this](const json& data) { handle_position_updates(data); });
}

void DeribitExchange::handle_private_trade_updates(const json& json_array) {
    for (size_t ii = 0; ii < json_array.size(); ++ii) {
        const auto& data = json_array[ii];
        if (data["instrument_name"] != this->symbol) {
            continue;
        }
        
        std::string trade_id = data["trade_id"];
        
        std::lock_guard<std::mutex> lock(this->fill_mutex);
        if (processed_trades.find(trade_id) != processed_trades.end()) {
            continue;  // Already processed
        }
        
        Order order;
        order.amount = data["amount"];
        order.is_taker = !data["post_only"];
        order.price = data["price"];
        order.side = caseInsensitiveCompare(data["direction"], "buy") ? OrderSide::BUY : OrderSide::SELL;
        order.state = OrderState::FILLED;
        order.orderId = data["order_id"];
        order.microSecond = data["timestamp"];
        
        processed_trades.insert(trade_id);
        this->executions.push_back(order);
    }
}

void DeribitExchange::handle_order_book_updates(const json& data) {
    size_t write_slot;
    auto& book = this->book_buffer.get_write_slot(write_slot);
    
    size_t idx = 0;
    for (const auto& bid : data["bids"]) {
        if (idx >= MAX_BOOK_LEVELS) break;
        book.bid_prices[idx] = bid[0].get<double>();
        book.bid_sizes[idx] = bid[1].get<double>();
        ++idx;
    }

    idx = 0;
    for (const auto& ask : data["asks"]) {
        if (idx >= MAX_BOOK_LEVELS) break;
        book.ask_prices[idx] = ask[0].get<double>();
        book.ask_sizes[idx] = ask[1].get<double>();
        ++idx;
    }

    this->book_buffer.commit_write(write_slot);
}

void DeribitExchange::handle_order_updates(const json& /*data*/) {
    // Order state updates are handled via trade callbacks
    // This callback is available for future order tracking if needed
}

void DeribitExchange::handle_position_updates(const json& /*data*/) {
    // Position updates can be handled here if needed
    // Currently positions are fetched via REST API
}

bool DeribitExchange::next_read(size_t& slot, OrderBook& book) {
    book = this->book_buffer.get_read_slot(slot);
    return true;
}

void DeribitExchange::fetchPosition(double& posAmount, double& avgPrice, bool is_hedge) {
    RESTApi.fetch_position(is_hedge ? hedge_symbol : symbol, posAmount, avgPrice);
}

void DeribitExchange::cancelOrders() {
    db_client.cancel_all_orders();
}

std::vector<Order> DeribitExchange::getFills() {
    std::lock_guard<std::mutex> lock(this->fill_mutex);
    std::vector<Order> fills;
    fills.swap(this->executions);
    return fills;
}

void DeribitExchange::quote(const std::string& /*order_id*/, OrderSide side, double price, double amount) {
    std::string sidestr = side == OrderSide::BUY ? "buy" : "sell";
    this->db_client.place_order(sidestr, price, amount, sidestr, false, "limit");
}

void DeribitExchange::market(const std::string& /*order_id*/, OrderSide side, double price, double amount, bool is_hedge) {
    std::string sidestr = side == OrderSide::BUY ? "buy" : "sell";
    this->db_client.place_order(sidestr, price, amount, sidestr, is_hedge, "market");
}

} // namespace RLTrader
