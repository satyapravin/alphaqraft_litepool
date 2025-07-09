#include "crypto_exchange.h"

#include <chrono>
#include <iostream>
#include <thread>

namespace RLTrader
{
/* ---------------- ctor / reset ------------------------------------ */
CryptoExchange::CryptoExchange(const std::string& symbol,
                               const std::string& hedge_symbol,
                               const std::string& api_key,
                               const std::string& api_secret)
    : client_(api_key, api_secret, symbol, hedge_symbol),
      rest_  (api_key, api_secret),
      symbol_(symbol),
      hedge_symbol_(hedge_symbol.empty() ? symbol : hedge_symbol)
{
    std::cout << "[CryptoExchange] constructed\n";
    set_callbacks();
    client_.start();
}

void CryptoExchange::reset()
{
    std::cout << "[CryptoExchange] reset\n";
    client_.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
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
    client_.set_orderbook_cb( [this](const json& j) { on_orderbook(j); } );
    client_.set_trade_cb    ( [this](const json& j) { on_private_trades(j); } );
}

void CryptoExchange::on_orderbook(const json& d)
{
    if (!d.contains("bids") || !d.contains("asks")) return;

    size_t slot{};
    auto&  book = book_buf_.get_write_slot(slot);

    size_t i = 0;
    for (const auto& b : d["bids"]) {
        if (i >= 20) break;
        book.bid_prices[i] = b.value("price",    0.0);
        book.bid_sizes [i] = b.value("quantity", 0.0);
        ++i;
    }
    i = 0;
    for (const auto& a : d["asks"]) {
        if (i >= 20) break;
        book.ask_prices[i] = a.value("price",    0.0);
        book.ask_sizes [i] = a.value("quantity", 0.0);
        ++i;
    }
    book_buf_.commit_write(slot);
}

void CryptoExchange::on_private_trades(const json& arr)
{
    std::lock_guard lk(fill_mtx_);
    if (!arr.is_array()) return;

    for (const auto& tr : arr) {
        const std::string tid = tr.value("trade_id", "");
        if (tid.empty() || !seen_trades_.insert(tid).second) continue;

        Order o;
        o.orderId     = tr.value("order_id", "");
        o.amount      = tr.value("quantity", 0.0);
        o.price       = tr.value("price",    0.0);
        o.side        = tr.value("side", "") == "BUY" ? OrderSide::BUY
                                                      : OrderSide::SELL;
        o.state       = OrderState::FILLED;
        o.microSecond = tr.value("create_time", 0L);

        executions_.push_back(o);
    }
}

/* ---------------- BaseExchange impl ------------------------------- */
bool CryptoExchange::next_read(size_t& slot, OrderBook& book)
{
    book  = book_buf_.get_read_slot(slot);
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

void CryptoExchange::cancelOrders() { client_.cancel_all_orders(); }

void CryptoExchange::quote(std::string oid, OrderSide s,
                           const double& price,const double& amt)
{
    client_.place_order(s == OrderSide::BUY ? "BUY" : "SELL",
                        price, amt, std::move(oid), false, "LIMIT");
}

void CryptoExchange::market(std::string oid, OrderSide s,
                            const double& price,const double& amt,bool hedge)
{
    client_.place_order(s == OrderSide::BUY ? "BUY" : "SELL",
                        price, amt, std::move(oid), hedge, "MARKET");
}

} // namespace RLTrader
