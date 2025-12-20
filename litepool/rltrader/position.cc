#include <cmath>
#include <stdexcept>
#include "position.h"

#include <iostream>
using namespace RLTrader;

Position::Position(BaseInstrument& instr, const double& aBalance,
    const double& initialQty, const double& initialAvgprice)
    :instrument(instr)
    , averagePrice(initialAvgprice)
    , netAmount(initialQty)
    , totalFee(0)
    , numOfTrades(0)
    , initialBalance(aBalance)
    , balance(aBalance){
}

void Position::reset(const double& initialQty, const double& initialPrice) {
    averagePrice = initialPrice;
    netAmount = initialQty;
    totalFee = 0.0;
    numOfTrades = 0;
    balance = initialBalance;
    trade_info.buy_trades = 0;
    trade_info.sell_trades = 0;
    trade_info.buy_amount = 0;
    trade_info.sell_amount = 0;
    trade_info.average_buy_price = 0;
    trade_info.average_sell_price = 0;
    
    // Clear LIFO stacks
    long_stack.clear();
    short_stack.clear();
    spreadCapture = 0.0;
}

PositionInfo Position::getPositionInfo(const double& bidPrice, const double& askPrice) const {
    PositionInfo info;
    double mid = 0.5 * (bidPrice + askPrice);
    info.balance = this->balance;
    info.inventoryPnL = this->inventoryPnL(mid);
    info.averagePrice = this->averagePrice;
    info.realizedPnL = this->balance - this->initialBalance;
    info.netPosition = this->instrument.getPositionFromAmount(netAmount, mid);
    info.leverage = 0;
    info.fees = totalFee;
    info.spreadCapture = this->spreadCapture;  // LIFO spread capture
    
    if (this->averagePrice > instrument.getTickSize()) {
        // Equity = balance + unrealized_pnl - fees
        double equity = balance + info.inventoryPnL - totalFee;
        info.leverage = this->instrument.getLeverage(netAmount, equity, mid);
    }

    return info;
}

double Position::inventoryPnL(const double& price) const {
    return this->instrument.pnl(netAmount, averagePrice, price);
}

void Position::onFill(const Order& order)
{
    if (order.state != OrderState::FILLED) {
        throw std::runtime_error("Order state not filled");
    }

    if (order.amount < 0) {
        throw std::runtime_error("Invalid order amount");
    }

    // Track trade info
    if (order.side == OrderSide::BUY) {
        trade_info.average_buy_price *= trade_info.buy_amount;
        trade_info.buy_trades++;
        trade_info.average_buy_price += order.price * order.amount;
        trade_info.buy_amount += order.amount;
        trade_info.average_buy_price /= trade_info.buy_amount;
    } else {
        trade_info.average_sell_price *= trade_info.sell_amount;
        trade_info.sell_trades++;
        trade_info.average_sell_price += order.price * order.amount;
        trade_info.sell_amount += order.amount;
        trade_info.average_sell_price /= trade_info.sell_amount;
    }

    // =========================================================================
    // LIFO Spread Capture: Track open positions and calculate spread on close
    // =========================================================================
    // BUY fills: close shorts (LIFO) or open new longs
    // SELL fills: close longs (LIFO) or open new shorts
    // =========================================================================
    
    double remaining_amount = order.amount;
    
    if (order.side == OrderSide::BUY) {
        // BUY: First try to close short positions (LIFO - from back)
        while (remaining_amount > 0 && !short_stack.empty()) {
            auto& entry = short_stack.back();
            double close_amount = std::min(remaining_amount, entry.amount);
            
            // Spread capture: we sold at entry.price (opened short), now buying back at order.price (closing short)
            // Use instrument.pnl() to handle both normal and inverse instruments correctly
            // For shorts: qty is negative, entryPrice is where we sold, exitPrice is where we buy back
            double spread = instrument.pnl(-close_amount, entry.price, order.price);
            spreadCapture += spread;
            
            remaining_amount -= close_amount;
            entry.amount -= close_amount;
            
            if (entry.amount <= 0) {
                short_stack.pop_back();
            }
        }
        
        // Remaining amount opens new long position
        if (remaining_amount > 0) {
            long_stack.push_back({OrderSide::BUY, order.price, remaining_amount});
        }
    } else {
        // SELL: First try to close long positions (LIFO - from back)
        while (remaining_amount > 0 && !long_stack.empty()) {
            auto& entry = long_stack.back();
            double close_amount = std::min(remaining_amount, entry.amount);
            
            // Spread capture: we bought at entry.price (opened long), now selling at order.price (closing long)
            // Use instrument.pnl() to handle both normal and inverse instruments correctly
            // For longs: qty is positive, entryPrice is where we bought, exitPrice is where we sell
            double spread = instrument.pnl(close_amount, entry.price, order.price);
            spreadCapture += spread;
            
            remaining_amount -= close_amount;
            entry.amount -= close_amount;
            
            if (entry.amount <= 0) {
                long_stack.pop_back();
            }
        }
        
        // Remaining amount opens new short position
        if (remaining_amount > 0) {
            short_stack.push_back({OrderSide::SELL, order.price, remaining_amount});
        }
    }

    // Original position tracking (for legacy compatibility)
    double pnl = 0;
    double sideSign = order.side == OrderSide::BUY ? 1.0 : -1.0;

    const double positionThreshold = instrument.getMinAmount() * 0.01;
    if (std::abs(netAmount) < positionThreshold) {
        averagePrice = order.price;
        netAmount = order.amount * sideSign;
    }
    else if (std::signbit(netAmount) == std::signbit(sideSign)) {
        averagePrice = (order.amount * order.price + std::abs(netAmount) * averagePrice) / (order.amount + std::abs(netAmount));
        netAmount += order.amount * sideSign;
    }
    else {
        if (order.amount >= std::abs(netAmount)) {
            pnl = instrument.pnl(netAmount, averagePrice, order.price);
            netAmount += order.amount * sideSign;
            averagePrice = std::abs(netAmount) > 0 ? order.price : 0.0;
        }
        else {
            pnl = instrument.pnl(order.amount * (netAmount > 0 ? 1.0 : -1.0), averagePrice, order.price);
            netAmount += order.amount * sideSign;
        }
    }

    totalFee += instrument.fees(order.amount, order.price, !order.is_taker);
    numOfTrades++;
    balance += pnl;
}
