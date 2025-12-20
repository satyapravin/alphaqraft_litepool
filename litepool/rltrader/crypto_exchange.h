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

#include "base_exchange.h"
#include "crypto_client.h"
#include "crypto_rest.h"
#include "orderbook_buffer.h"

#include <atomic>
#include <mutex>
#include <unordered_set>

namespace RLTrader
{
    // Simple adapter gluing CryptoClient + CryptoREST to BaseExchange
    class CryptoExchange final : public BaseExchange
    {
    public:
        CryptoExchange(const std::string& symbol,
                       const std::string& hedge_symbol,
                       const std::string& api_key,
                       const std::string& api_secret);

        /* BaseExchange ------------------------------------------------ */
        void reset() override;
        bool next_read(size_t& slot, OrderBook& book) override;
        void done_read(size_t slot) override { book_buf_.commit_read(slot); }
        void toBook(const std::unordered_map<std::string,double>&,OrderBook&) override
        { throw std::runtime_error("Not implemented"); }
        void fetchPosition(double& a,double& p,bool hedge) override;
        std::vector<Order> getFills() override;
        void cancelOrders() override;
        bool isDummy() override { return false; }

        const std::map<std::string,Order>& getBidOrders() const override
        { throw std::runtime_error("N/A"); }
        const std::map<std::string,Order>& getAskOrders() const override
        { throw std::runtime_error("N/A"); }
        std::vector<Order> getUnackedOrders() const override
        { throw std::runtime_error("N/A"); }

        void quote(const std::string& order_id, OrderSide side, double price, double amount) override;
        void market(const std::string& order_id, OrderSide side, double price, double amount, bool hedge) override;

    private:
        /* internal helpers ------------------------------------------- */
        void set_callbacks();
        void on_orderbook     (const json& d);
        void on_private_trades(const json& d);

        CryptoClient            client_;
        CryptoREST             rest_;
        LockFreeOrderBookBuffer book_buf_;

        std::mutex                    fill_mtx_;
        std::vector<Order>            executions_;
        std::unordered_set<std::string> seen_trades_;

        const std::string symbol_;
        const std::string hedge_symbol_;
    };
} // namespace RLTrader
