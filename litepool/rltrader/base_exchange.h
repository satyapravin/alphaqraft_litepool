#pragma once
#include <map>
#include <vector>
#include <unordered_map>
#include "order.h"
#include "orderbook.h"

namespace RLTrader {
    class BaseExchange {
    public:
        // Constructor
        virtual ~BaseExchange() = default;

        // Resets the exchange's state
        virtual void reset() = 0;

        // Advances to the next row in the data
        virtual bool next_read(size_t& slot, OrderBook& book) = 0;

        virtual void done_read(size_t slot) = 0;

        // build order book from labeled data
        virtual void toBook(const std::unordered_map<std::string, double>& lob, OrderBook &book) = 0;

        // fetches the current position from exchange
        virtual void fetchPosition(double& posAmount, double& avgPrice, bool is_hedge) = 0;

        // Returns executed orders and clears them
        virtual std::vector<Order> getFills() = 0;

        // Processes order cancellation
        virtual void cancelOrders() = 0;

	virtual bool isDummy() = 0;

        [[nodiscard]] virtual const std::map<std::string, Order>& getBidOrders() const = 0;

        [[nodiscard]] virtual const std::map<std::string, Order>& getAskOrders() const = 0;

        [[nodiscard]] virtual std::vector<Order> getUnackedOrders() const = 0;

        virtual void quote(std::string order_id, OrderSide side, const double& price, const double& amount) = 0;

        virtual void market(std::string order_id, OrderSide side, const double& price, const double& amount, bool hedge) = 0;
    };
}
