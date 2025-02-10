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
}

PositionInfo Position::getPositionInfo(const double& bidPrice, const double& askPrice) const {
    PositionInfo info;
    double mid = 0.5 * (bidPrice + askPrice);
    info.balance = this->balance;
    info.inventoryPnL = this->inventoryPnL(mid);
    info.averagePrice = this->averagePrice;
    info.tradingPnL = this->balance - this->initialBalance;
    info.netPosition = this->instrument.getPositionFromAmount(netAmount, mid);
    info.leverage = 0;
    info.fees = totalFee;
    
    if (this->averagePrice > instrument.getTickSize()) {
        info.leverage = this->instrument.getLeverage(netAmount, balance + info.inventoryPnL, mid);
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

    double pnl = 0;
    double sideSign = order.side == OrderSide::BUY ? 1.0 : -1.0;

    if (std::abs(netAmount) < 0.00000001) {
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
