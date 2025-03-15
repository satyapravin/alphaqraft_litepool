#include "env_adaptor.h"
#include <algorithm>
#include <iostream>
#include <algorithm>  

using namespace RLTrader;

EnvAdaptor::EnvAdaptor(Strategy& strat, BaseExchange& exch):
            strategy(strat),
            exchange(exch),
            market_builder(std::make_unique<MarketSignalBuilder>()),
            position_builder(std::make_unique<PositionSignalBuilder>()),
            trade_builder(std::unique_ptr<TradeSignalBuilder>()),
            bid_prices(), ask_prices(), bid_sizes(), ask_sizes() {
}

bool EnvAdaptor::next() {
    std::fill_n(state.begin(), 242*5, 0);
    for (int ii=0; ii < 5; ++ii) {
        OrderBook book;
        size_t read_slot;
        if(this->exchange.next_read(read_slot, book)) {
            this->strategy.next();
            computeState(book);

           
            std::copy(book.bid_prices.begin(), book.bid_prices.end(), bid_prices.begin());
            std::copy(book.ask_prices.begin(), book.ask_prices.end(), ask_prices.begin());
            std::copy(book.bid_sizes.begin(),  book.bid_sizes.end(),  bid_sizes.begin());
            std::copy(book.ask_sizes.begin(),  book.ask_sizes.end(),  ask_sizes.begin());
            this->exchange.done_read(read_slot);
        } else {
            return false;
        }
    }
    
    return true;
}

void EnvAdaptor::getState(std::array<double, 242*5>& st) {
    st = state;
}

void EnvAdaptor::quote(int buy_spread, int sell_spread, const double& buy_percent, const double& sell_percent) {
    this->strategy.quote(buy_spread, sell_spread, buy_percent, sell_percent, bid_prices, ask_prices);
}

void EnvAdaptor::reset() {
    max_realized_pnl = 0;
    max_unrealized_pnl = 0;
    drawdown = 0;
    auto market_ptr = std::make_unique<MarketSignalBuilder>();
    market_builder = std::move(market_ptr);
    auto position_ptr = std::make_unique<PositionSignalBuilder>();
    position_builder = std::move(position_ptr);
    auto trade_ptr = std::make_unique<TradeSignalBuilder>();
    trade_builder = std::move(trade_ptr);
    this->strategy.reset();
    std::fill_n(state.begin(), 242*5, 0);
}


void EnvAdaptor::getInfo(std::unordered_map<std::string, double>& inf) {
    inf = std::move(info);
}

void EnvAdaptor::computeInfo(OrderBook &book) {
    auto bid_price = book.bid_prices[0];
    auto ask_price = book.ask_prices[0];
    PositionInfo posInfo =  strategy.getPosition().getPositionInfo(bid_price, ask_price);
    auto tradeInfo = strategy.getPosition().getTradeInfo();

    if (max_unrealized_pnl < posInfo.inventoryPnL) max_unrealized_pnl = posInfo.inventoryPnL;
    if (max_realized_pnl < posInfo.tradingPnL) max_realized_pnl = posInfo.tradingPnL;
    double latest_dd = std::min(posInfo.inventoryPnL - max_unrealized_pnl, 0.0) + std::min(posInfo.tradingPnL - max_realized_pnl, 0.0);
    if (drawdown > latest_dd) drawdown = latest_dd;
    info.clear();
    info["mid_price"] = (bid_price + ask_price) * 0.5;
    info["balance"] = posInfo.balance;
    info["unrealized_pnl"] = posInfo.inventoryPnL;
    info["realized_pnl"] = posInfo.tradingPnL;
    info["leverage"] = posInfo.leverage;
    info["trade_count"] = static_cast<double>(tradeInfo.buy_trades + tradeInfo.sell_trades);
    info["drawdown"] = drawdown;
    info["fees"] = posInfo.fees;
    info["delta_amount"] = tradeInfo.buy_amount - tradeInfo.sell_amount;
}


void EnvAdaptor::computeState(OrderBook& book)
{
    auto bid_price = book.bid_prices[0];
    auto ask_price = book.ask_prices[0];
    auto market_signals = market_builder->add_book(book);
    PositionInfo position_info = strategy.getPosition().getPositionInfo(book.bid_prices[0], book.ask_prices[0]);
    if (position_info.inventoryPnL > max_unrealized_pnl) max_unrealized_pnl = position_info.inventoryPnL;
    if (position_info.tradingPnL > max_realized_pnl) max_realized_pnl = position_info.tradingPnL;
    auto position_signals = position_builder->add_info(position_info, bid_price, ask_price);
    TradeInfo trade_info = strategy.getPosition().getTradeInfo();
    auto trade_signals = trade_builder->add_trade(trade_info, bid_price, ask_price);
    
    std::copy_n(state.begin(), market_signals.size(), market_signals.begin());
    std::copy_n(state.begin() + market_signals.size(), position_signals.size(), position_signals.begin());
    std::copy_n(state.begin() + market_signals.size() + position_signals.size(), trade_signals.size(), market_signals.begin());
    computeInfo(book);
}
