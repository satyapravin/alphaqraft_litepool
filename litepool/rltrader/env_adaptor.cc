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
    std::fill_n(state.begin(), 242*2, 0);
    for (int ii=0; ii < 2; ++ii) {
        OrderBook book;
        size_t read_slot;
        if(this->exchange.next_read(read_slot, book)) {
            this->strategy.next();
            computeState(ii, book);

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

void EnvAdaptor::getState(std::array<double, 242*2>& st) {
    st = state;
}

void EnvAdaptor::quote(const double& bid_spread, const double& ask_spread, const double& bid_size, const double& ask_size) {
    this->strategy.quote(bid_spread, ask_spread, bid_size, ask_size, bid_prices, ask_prices);
}

void EnvAdaptor::reset() {
    std::cout << "Reset env" << std::endl;
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
    std::fill_n(state.begin(), 242*2, 0);
    mid_price_deque.clear();
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
    auto mid = (bid_price + ask_price) * 0.5;
    info["mid_price"] = mid;
    mid_price_deque.push_back(mid);
    mid -= mid_price_deque.front(); 
    if (mid_price_deque.size() > 1) { mid_price_deque.pop_front(); }
    info["mid_diff"] = mid;
    info["balance"] = posInfo.balance;
    info["unrealized_pnl"] = posInfo.inventoryPnL;
    info["realized_pnl"] = posInfo.tradingPnL;
    info["leverage"] = posInfo.leverage;
    info["trade_count"] = static_cast<double>(tradeInfo.buy_trades + tradeInfo.sell_trades);
    info["drawdown"] = drawdown;
    info["fees"] = posInfo.fees;
    info["average_price"] = posInfo.averagePrice;
}


void EnvAdaptor::computeState(size_t ii, OrderBook& book)
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
    size_t state_begin = ii * 242;
    std::copy_n(market_signals.begin(), market_signals.size(), state.begin() + state_begin);
    std::copy_n(position_signals.begin(), position_signals.size(), state.begin() + state_begin + market_signals.size());
    std::copy_n(trade_signals.begin(), trade_signals.size(), state.begin() + state_begin + market_signals.size() + position_signals.size());
    computeInfo(book);
}
