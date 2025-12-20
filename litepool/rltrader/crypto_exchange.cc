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

#include "crypto_exchange.h"

#include <chrono>
#include <thread>

namespace RLTrader {

namespace {
    constexpr size_t MAX_BOOK_LEVELS = 20;
    constexpr auto RESET_DELAY = std::chrono::milliseconds(500);
    
    // Safe string to double conversion
    double safeStod(const std::string& str, double default_val = 0.0) {
        try {
            return std::stod(str);
        } catch (...) {
            return default_val;
        }
    }
}

/* ---------------- ctor / reset ------------------------------------ */
CryptoExchange::CryptoExchange(const std::string& symbol,
                               const std::string& hedge_symbol,
                               const std::string& api_key,
                               const std::string& api_secret)
    : client_(api_key, api_secret, symbol, hedge_symbol),
      rest_(api_key, api_secret),
      symbol_(symbol),
      hedge_symbol_(hedge_symbol.empty() ? symbol : hedge_symbol)
{
    set_callbacks();
    client_.start();
}

void CryptoExchange::reset()
{
    client_.stop();
    std::this_thread::sleep_for(RESET_DELAY);
    {
        std::lock_guard lk(fill_mtx_);
        executions_.clear();
        seen_trades_.clear();
    }
    client_.start();
}

/* ---------------- callbacks --------------------------------------- */
void CryptoExchange::set_callbacks()
{
    client_.set_orderbook_cb([this](const json& j) { on_orderbook(j); });
    client_.set_trade_cb([this](const json& j) { on_private_trades(j); });
}

void CryptoExchange::on_orderbook(const json& data)
{
    if (data.empty() || !data[0].contains("bids") || !data[0].contains("asks")) {
        return;  // Invalid data, skip silently
    }

    size_t slot{};
    auto& book = book_buf_.get_write_slot(slot);

    size_t i = 0;
    for (const auto& b : data[0]["bids"]) {
        if (i >= MAX_BOOK_LEVELS || !b.is_array() || b.size() < 2) break;
        book.bid_prices[i] = safeStod(b[0].get<std::string>());
        book.bid_sizes[i] = safeStod(b[1].get<std::string>());
        ++i;
    }
    
    i = 0;
    for (const auto& a : data[0]["asks"]) {
        if (i >= MAX_BOOK_LEVELS || !a.is_array() || a.size() < 2) break;
        book.ask_prices[i] = safeStod(a[0].get<std::string>());
        book.ask_sizes[i] = safeStod(a[1].get<std::string>());
        ++i;
    }

    book_buf_.commit_write(slot);
}

void CryptoExchange::on_private_trades(const json& d)
{
    if (!d.contains("data")) return;
    
    std::lock_guard lk(fill_mtx_);

    for (const auto& tr : d["data"]) {
        const std::string tid = tr.value("trade_id", "");
        if (tid.empty() || !seen_trades_.insert(tid).second) continue;
        if (tr.value("instrument_name", "") != symbol_) continue;
        
        Order o;
        o.orderId = tr.value("order_id", "");
        o.amount = std::abs(safeStod(tr.value("traded_quantity", "0.0")));
        o.price = std::abs(safeStod(tr.value("traded_price", "0.0")));
        o.side = tr.value("side", "") == "BUY" ? OrderSide::BUY : OrderSide::SELL;
        o.state = OrderState::FILLED;
        o.is_taker = true;  // Assume taker for private trades
        o.microSecond = tr.value("create_time", 0LL);
        
        executions_.push_back(o);
    }
}

/* ---------------- BaseExchange impl ------------------------------- */
bool CryptoExchange::next_read(size_t& slot, OrderBook& book)
{
    book = book_buf_.get_read_slot(slot);
    return (book.bid_prices[0] != 0.0 || book.ask_prices[0] != 0.0);
}

void CryptoExchange::fetchPosition(double& a, double& p, bool hedge)
{
    rest_.fetch_position(hedge ? hedge_symbol_ : symbol_, a, p);
}

std::vector<Order> CryptoExchange::getFills()
{
    std::lock_guard lk(fill_mtx_);
    std::vector<Order> out;
    out.swap(executions_);
    return out;
}

void CryptoExchange::cancelOrders() 
{ 
    client_.cancel_all_orders(); 
}

void CryptoExchange::quote(const std::string& order_id, OrderSide side, double price, double amount)
{
    client_.place_order(side == OrderSide::BUY ? "BUY" : "SELL",
                        price, amount, order_id, false, "LIMIT");
}

void CryptoExchange::market(const std::string& order_id, OrderSide side, double price, double amount, bool hedge)
{
    client_.place_order(side == OrderSide::BUY ? "BUY" : "SELL",
                        price, amount, order_id, hedge, "MARKET");
}

} // namespace RLTrader
