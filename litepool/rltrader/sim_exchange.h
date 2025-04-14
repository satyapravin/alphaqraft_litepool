#pragma once
#include <string>
#include <map>
#include <vector>
#include <cstdint>

#include "base_exchange.h"
#include "order.h"
#include "csv_reader.h"
#include "orderbook.h"

namespace RLTrader {
    class SimExchange final : public BaseExchange {
    public:
        // initialized labels
        static bool initialize();

        // Constructor
        SimExchange(const std::string& filename, long delay, int start_read, int max_read); // 300 milliseconds delay

        // Generates an OrderBook
        void toBook(const std::unordered_map<std::string, double>& lob, OrderBook& book) override;

        // Resets the exchange's state
        void reset() override;
       
        bool isDummy() override { return true; }

        // Advances to the next row in the data
        bool next_read(size_t& slot, OrderBook& book) override;

        void done_read(size_t slot) override;

        // fetch dummy zero positions
        void fetchPosition(double& posAmount, double& avgPrice, bool is_hedge) override;


        // Returns executed orders and clears them
        std::vector<Order> getFills() override;

        // Processes order cancellation
        void cancelOrders() override;

         const std::map<std::string, Order>& getBidOrders() const override;

         const std::map<std::string, Order>& getAskOrders() const override;

         std::vector<Order> getUnackedOrders() const override;

         void quote(std::string order_id, OrderSide side, const double& price, const double& amount) override;

         void market(std::string order_id, OrderSide side, const double& price, const double& amount, bool hedge) override;

    private:
        CsvReader dataReader; // reader
        long delay;           // Delay to process timed buffer  
        std::map<std::string, Order> bid_quotes;  // Active buy orders
        std::map<std::string, Order> ask_quotes;  // Active sell orders
        std::vector<Order> executions;       // Executed orders
        std::map<long long, std::vector<Order>> timed_buffer;  // Orders waiting for processing

        // cancel orders for a side
        void cancel(std::map<std::string, Order>& quotes);

        // Executes orders based on current market conditions
        void execute();

        // Adds orders to the buffer
        void addToBuffer(const Order& order);

        // Processes orders that are pending based on their timestamps
        void processPending(const DataRow& obs);

        static std::vector<std::string> ask_price_labels;
        static std::vector<std::string> bid_price_labels;
        static std::vector<std::string> ask_size_labels;
        static std::vector<std::string> bid_size_labels;
        static bool init;
    };
}
