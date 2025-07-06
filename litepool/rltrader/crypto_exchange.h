#pragma once
#include "base_exchange.h"
#include "crypto_client.h"
#include "crypto_rest.h"
#include "orderbook_buffer.h"
#include <mutex>
#include <unordered_set>
#include <atomic>

namespace RLTrader {
    // Thread safety:
    // - All public methods are thread-safe.
    // - fill_mtx_ protects executions_ and seen_trades_.
    // - Callbacks (on_orderbook, on_private_trades) are invoked by CryptoClient in separate threads,
    //   but fill_mtx_ ensures thread-safe access to shared data.
    // - reset() stops and restarts CryptoClient, safe to call concurrently.
    class CryptoExchange final : public BaseExchange {
    public:
        CryptoExchange(const std::string& symbol,
                       const std::string& hedge_symbol,
                       const std::string& api_key,
                       const std::string& api_secret);

        /* BaseExchange interface ----------------------------------------- */
        void reset() override;
        bool next_read(size_t& slot, OrderBook& book) override;
        void done_read(size_t slot) override { book_buf_.commit_read(slot); }
        void toBook(const std::unordered_map<std::string,double>&, OrderBook&) override {
            throw std::runtime_error("Not implemented");
        }
        void fetchPosition(double& a, double& p, bool is_hedge) override;
        std::vector<Order> getFills() override;
        void cancelOrders() override;
        bool isDummy() override { return false; }

        const std::map<std::string,Order>& getBidOrders() const override { throw std::runtime_error("N/A"); }
        const std::map<std::string,Order>& getAskOrders() const override { throw std::runtime_error("N/A"); }
        std::vector<Order> getUnackedOrders() const override { throw std::runtime_error("N/A"); }

        void quote(std::string oid, OrderSide side, const double& price, const double& amt) override;
        void market(std::string oid, OrderSide side, const double& price, const double& amt, bool hedge) override;

    private:
        void set_callbacks();
        void on_private_trades(const json& d);
        void on_orderbook(const json& d);

        CryptoClient client_;
        CryptoREST rest_;
        LockFreeOrderBookBuffer book_buf_;

        std::mutex fill_mtx_;
        std::vector<Order> executions_;
        std::unordered_set<std::string> seen_trades_;

        const std::string symbol_;
        const std::string hedge_symbol_;
    };
}
