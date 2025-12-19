#pragma once
#include "base_instrument.h"
#include "order.h"
#include <deque>

namespace RLTrader {
	struct PositionInfo {
        double netPosition = 0;
        double balance = 0;
        double averagePrice = 0;
        double realizedPnL = 0;  // Gross realized PnL (before fees)
        double inventoryPnL = 0;
        double leverage = 0;
	double fees = 0;
        double spreadCapture = 0;  // Cumulative spread captured from closed round-trips
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
        double spreadCapture = 0.0;         // Cumulative spread captured

    public:
        Position(BaseInstrument& instr, const double& aBalance, const double& initialQty, const double& initialAvgprice);
        void reset(const double& initialQty, const double& initialAvgprice);
        [[nodiscard]] PositionInfo getPositionInfo(const double& bidPrice, const double& askPrice) const;
        void onFill(const Order& order);
        [[nodiscard]] double inventoryPnL(const double& price) const;
        [[nodiscard]] double getNetAmount() const { return netAmount; }
        [[nodiscard]] double getInitialBalance() const { return initialBalance; }
        [[nodiscard]] long getNumberOfTrades() const { return numOfTrades; }
        [[nodiscard]] double getSpreadCapture() const { return spreadCapture; }
        TradeInfo& getTradeInfo() { return trade_info; }
    };
}
