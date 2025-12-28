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
#include "base_instrument.h"
#include "order.h"
#include <deque>

namespace RLTrader {
	struct PositionInfo {
        double netPosition = 0;
        double balance = 0;
        double averagePrice = 0;
        double realizedPnL = 0;  // LIFO-based realized PnL from closed positions (same as spreadCapture)
        double inventoryPnL = 0;      // Average cost unrealized P&L
        double lifoUnrealizedPnL = 0; // LIFO-based unrealized P&L (from remaining stack entries)
        double leverage = 0;
	double fees = 0;
        double spreadCapture = 0;  // Cumulative spread captured from closed round-trips (LIFO)
    };

    struct TradeInfo {
        long buy_trades = 0;
        long sell_trades = 0;
        double buy_amount = 0;
        double sell_amount = 0;
        double average_buy_price = 0;
        double average_sell_price = 0;
    };

    // Individual open position entry for LIFO tracking
    struct OpenEntry {
        OrderSide side;
        double price;
        double amount;
    };

    class Position {
    private:
        BaseInstrument& instrument;
        double averagePrice = 0.0;
        double netAmount = 0.0;
        double totalFee = 0.0;
        long numOfTrades = 0;
        double initialBalance = 0.0;
        double balance = 0.0;
        TradeInfo trade_info;

        // LIFO stacks for spread capture tracking
        std::deque<OpenEntry> long_stack;   // BUY entries waiting to close
        std::deque<OpenEntry> short_stack;  // SELL entries waiting to close
        double spreadCapture = 0.0;         // Cumulative spread captured (LIFO, for rewards)
        double realizedPnL = 0.0;           // Cumulative realized PnL (weighted-average, for logging/cash flow)

    public:
        Position(BaseInstrument& instr, const double& aBalance, const double& initialQty, const double& initialAvgprice);
        void reset(const double& initialQty, const double& initialAvgprice);
        [[nodiscard]] PositionInfo getPositionInfo(const double& bidPrice, const double& askPrice) const;
        void onFill(const Order& order);
        [[nodiscard]] double inventoryPnL(const double& price) const;
        [[nodiscard]] double lifoUnrealizedPnL(const double& price) const;  // LIFO-based unrealized from remaining stack entries
        [[nodiscard]] double getNetAmount() const { return netAmount; }
        [[nodiscard]] double getInitialBalance() const { return initialBalance; }
        [[nodiscard]] long getNumberOfTrades() const { return numOfTrades; }
        [[nodiscard]] double getSpreadCapture() const { return spreadCapture; }
        TradeInfo& getTradeInfo() { return trade_info; }
    };
}
