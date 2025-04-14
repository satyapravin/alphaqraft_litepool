#pragma once
#include <mutex>
#include <unordered_set>
#include "base_exchange.h"
#include "deribit_client.h"
#include "deribit_rest.h"
#include "orderbook_buffer.h"

namespace RLTrader {
    class DeribitExchange final : public BaseExchange {
    public:
        // Constructor
        DeribitExchange(const std::string& symbol, const std::string& hedge_symbol, const std::string& api_key, const std::string& api_secret);

        // Generates an OrderBook
        void toBook(const std::unordered_map<std::string, double>& lob, OrderBook &book) override;

        // Resets the exchange's state
        void reset() override;

	bool isDummy() override { return false; }

        // Advances to the next row in the data
        bool next_read(size_t& slot, OrderBook& book) override;

        void done_read(size_t slot) override { this->book_buffer.commit_read(slot); }

        // fetch dummy zero positions
        void fetchPosition(double& posAmount, double& avgPrice, bool is_hedge) override;

        // Returns executed orders and clears them
        std::vector<Order> getFills() override;

        // Processes order cancellation
        void cancelOrders() override;

        [[nodiscard]] const std::map<std::string, Order>& getBidOrders() const override {
            throw std::runtime_error("Unimplemented");
        }

        [[nodiscard]] const std::map<std::string, Order>& getAskOrders() const override {
            throw std::runtime_error("Unimplemented");
        }

        [[nodiscard]] std::vector<Order> getUnackedOrders() const override {
            throw std::runtime_error("Unimplemented");
        }

        void quote(std::string order_id, OrderSide side, const double& price, const double& amount) override;

        void market(std::string order_id, OrderSide side, const double& price, const double& amount, bool is_hedge) override;

    private:
        void set_callbacks();
        void handle_private_trade_updates (const json& data);
        void handle_order_book_updates (const json& data);
        void handle_order_updates (const json& data);
        void handle_position_updates (const json& data);

        DeribitClient db_client;
        DeribitREST RESTApi;
        std::vector<Order> executions;

        double position_amount = 0;
        double position_avg_price = 0;
        LockFreeOrderBookBuffer book_buffer;
        std::string symbol;
        std::string hedge_symbol;
        std::mutex fill_mutex;
	std::unordered_set<std::string> processed_trades;
        std::atomic<long> orders_count;
    };

} // RLTrader
