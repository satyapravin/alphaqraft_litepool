#pragma once
#include "base_instrument.h"
#include "order.h"

namespace RLTrader {
	struct PositionInfo {
        double netPosition = 0;
        double balance = 0;
        double averagePrice = 0;
        double realizedPnL = 0;  // Gross realized PnL (before fees)
        double inventoryPnL = 0;
        double leverage = 0;
	double fees = 0;
    };

    struct TradeInfo {
        long buy_trades = 0;
        long sell_trades = 0;
        double buy_amount = 0;
        double sell_amount = 0;
        double average_buy_price = 0;
        double average_sell_price = 0;
    };

    class Position {
    private:
        BaseInstrument& instrument;
        double averagePrice = 0.0;
        double netAmount = 0.0;
        double totalFee = 0.0;
        int numOfTrades = 0;
        double initialBalance = 0.0;
        double balance = 0.0;
        TradeInfo trade_info;


    public:
        Position(BaseInstrument& instr, const double& aBalance, const double& initialQty, const double& initialAvgprice);
        void reset(const double& initialQty, const double& initialAvgprice);
        [[nodiscard]] PositionInfo getPositionInfo(const double& bidPrice, const double& askPrice) const;
        void onFill(const Order& order);
        [[nodiscard]] double inventoryPnL(const double& price) const;
        [[nodiscard]] double getNetAmount() const { return netAmount; }
        [[nodiscard]] double getInitialBalance() const { return initialBalance; }
        [[nodiscard]] long getNumberOfTrades() const { return numOfTrades; }
        TradeInfo& getTradeInfo() { return trade_info; }
    };
}
