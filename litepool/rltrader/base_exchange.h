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

        virtual void quote(const std::string& order_id, OrderSide side, double price, double amount) = 0;

        virtual void market(const std::string& order_id, OrderSide side, double price, double amount, bool hedge) = 0;
    };
}
