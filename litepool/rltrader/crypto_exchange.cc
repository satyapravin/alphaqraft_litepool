#include "crypto_exchange.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace RLTrader {

/* ---------------- ctor / reset -------------------------------------- */
CryptoExchange::CryptoExchange(const std::string& symbol,
                               const std::string& hedge_symbol,
                               const std::string& api_key,
                               const std::string& api_secret)
    : client_(api_key, api_secret, symbol, hedge_symbol),
      rest_(api_key, api_secret),
      symbol_(symbol),
      hedge_symbol_(hedge_symbol.empty() ? symbol : hedge_symbol) {
    set_callbacks();
    client_.start();
}

void CryptoExchange::reset() {
    std::cout << "[CryptoExchange] Resetting exchange\n";
    client_.stop();
    // Increased delay to ensure clean resource cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(15000)); // 15s delay
    executions_.clear();
    seen_trades_.clear();
    client_.start();
    std::cout << "[CryptoExchange] Done Reset exchange\n";
}

/* ---------------- callbacks ----------------------------------------- */
void CryptoExchange::set_callbacks() {
    client_.set_orderbook_cb([this](const json& d) { on_orderbook(d); });
    client_.set_trade_cb([this](const json& d) { on_private_trades(d); });
}

void CryptoExchange::on_orderbook(const json& d) {
    if (!d.is_array() || d.empty()) {
        std::cerr << "[CryptoExchange] Invalid orderbook data\n";
        return;
    }
    size_t slot;
    auto& book = book_buf_.get_write_slot(slot);
    size_t i = 0;
    for (const auto& bid : d["bids"]) {
        if (!bid.contains("price") || !bid.contains("quantity")) {
            std::cerr << "[CryptoExchange] Invalid bid data\n";
            continue;
        }
        book.bid_prices[i] = bid["price"].get<double>();
        book.bid_sizes[i] = bid["quantity"].get<double>();
        if (++i >= 20) break;
    }
    i = 0;
    for (const auto& ask : d["asks"]) {
        if (!ask.contains("price") || !ask.contains("quantity")) {
            std::cerr << "[CryptoExchange] Invalid ask data\n";
            continue;
        }
        book.ask_prices[i] = ask["price"].get<double>();
        book.ask_sizes[i] = ask["quantity"].get<double>();
        if (++i >= 20) break;
    }
    book_buf_.commit_write(slot);
}

void CryptoExchange::on_private_trades(const json& arr) {
    std::lock_guard lk(fill_mtx_);
    if (!arr.is_array()) {
        std::cerr << "[CryptoExchange] Invalid trades data\n";
        return;
    }
    for (const auto& tr : arr) {
        if (!tr.contains("trade_id") || !tr.contains("order_id") || !tr.contains("quantity") ||
            !tr.contains("price") || !tr.contains("side") || !tr.contains("create_time")) {
            std::cerr << "[CryptoExchange] Incomplete trade data\n";
            continue;
        }
        std::string tid = tr["trade_id"].get<std::string>();
        if (seen_trades_.insert(tid).second) {
            Order o;
            o.orderId = tr["order_id"].get<std::string>();
            o.amount = tr["quantity"].get<double>();
            o.price = tr["price"].get<double>();
            o.side = tr["side"] == "BUY" ? OrderSide::BUY : OrderSide::SELL;
            o.state = OrderState::FILLED;
            o.microSecond = tr["create_time"].get<long>();
            executions_.push_back(o);
            std::cout << "[CryptoExchange] Trade added: orderId=" << o.orderId << ", amount=" << o.amount
                      << ", price=" << o.price << "\n";
        }
    }
}

/* ---------------- BaseExchange impl --------------------------------- */
bool CryptoExchange::next_read(size_t& slot, OrderBook& book) {
    book = book_buf_.get_read_slot(slot);
    bool valid = book.bid_prices[0] != 0.0 || book.ask_prices[0] != 0.0;
    if (!valid) {
        std::cerr << "[CryptoExchange] Invalid order book data\n";
    }
    return valid;
}

void CryptoExchange::fetchPosition(double& a, double& p, bool is_hedge) {
    std::cout << "[CryptoExchange] Fetching position for " << (is_hedge ? hedge_symbol_ : symbol_) << "\n";
    bool success = rest_.fetch_position(is_hedge ? hedge_symbol_ : symbol_, a, p);
    std::cout << "[CryptoExchange] Position fetched: amount=" << a << ", avg_price=" << p
              << ", success=" << (success ? "true" : "false") << "\n";
    if (!success) {
        std::cerr << "[CryptoExchange] Warning: Failed to fetch position, possible REST API failure\n";
        a = p = 0.0; // Ensure safe defaults
    }
}

std::vector<Order> CryptoExchange::getFills() {
    std::lock_guard lk(fill_mtx_);
    std::vector<Order> tmp;
    tmp.swap(executions_);
    std::cout << "[CryptoExchange] Retrieved " << tmp.size() << " fills\n";
    return tmp;
}

void CryptoExchange::cancelOrders() {
    client_.cancel_all_orders();
}

void CryptoExchange::quote(std::string oid, OrderSide s, const double& pr, const double& amt) {
    client_.place_order(s == OrderSide::BUY ? "BUY" : "SELL", pr, amt, oid, false, "LIMIT");
}

void CryptoExchange::market(std::string oid, OrderSide s, const double& pr, const double& amt, bool hedge) {
    client_.place_order(s == OrderSide::BUY ? "BUY" : "SELL", pr, amt, oid, hedge, "MARKET");
}

} // namespace RLTrader
