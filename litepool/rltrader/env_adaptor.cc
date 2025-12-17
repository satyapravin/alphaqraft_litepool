#include "env_adaptor.h"
#include <algorithm>

using namespace RLTrader;

EnvAdaptor::EnvAdaptor(Strategy& strat, BaseExchange& exch, int ticks_per_step):
            strategy(strat),
            exchange(exch),
            ticks_per_step_(ticks_per_step),
            market_builder(std::make_unique<MarketSignalBuilder>()),
            bid_prices(), ask_prices(), bid_sizes(), ask_sizes() {
}

bool EnvAdaptor::next() {
    std::fill_n(state.begin(), state.size(), 0);
    OrderBook book;
    size_t read_slot;
    
    // Advance multiple ticks per RL step to let orders persist
    for (int tick = 0; tick < ticks_per_step_; ++tick) {
        if (!this->exchange.next_read(read_slot, book)) {
            return false;  // No more data
        }
        this->strategy.next();  // Process any fills from this tick
        this->exchange.done_read(read_slot);
    }
    
    // Compute state from the last tick
    computeState(book);
    std::copy(book.bid_prices.begin(), book.bid_prices.end(), bid_prices.begin());
    std::copy(book.ask_prices.begin(), book.ask_prices.end(), ask_prices.begin());
    std::copy(book.bid_sizes.begin(),  book.bid_sizes.end(),  bid_sizes.begin());
    std::copy(book.ask_sizes.begin(),  book.ask_sizes.end(),  ask_sizes.begin());
    
    return true;
}

void EnvAdaptor::getState(std::array<double, OBS_DIM>& st) {
    st = state;
}

void EnvAdaptor::quote(const RLAction& action) {
    this->strategy.quote(action, bid_prices, ask_prices);
}

void EnvAdaptor::reset() {
    max_realized_pnl = 0;
    max_unrealized_pnl = 0;
    drawdown = 0;
    auto market_ptr = std::make_unique<MarketSignalBuilder>();
    market_builder = std::move(market_ptr);
    this->strategy.reset();
    std::fill_n(state.begin(), state.size(), 0);
    mid_price_deque.clear();
    // Reset AMM simulator so it auto-initializes on first step with valid price
    amm_simulator.clear();
}


void EnvAdaptor::getInfo(std::unordered_map<std::string, double>& inf) {
    inf = info;  // Copy, don't move - we need to preserve info for terminal caching
}

void EnvAdaptor::computeInfo(OrderBook &book) {
    auto bid_price = book.bid_prices[0];
    auto ask_price = book.ask_prices[0];
    PositionInfo posInfo =  strategy.getPosition().getPositionInfo(bid_price, ask_price);
    auto tradeInfo = strategy.getPosition().getTradeInfo();

    if (max_unrealized_pnl < posInfo.inventoryPnL) max_unrealized_pnl = posInfo.inventoryPnL;
    if (max_realized_pnl < posInfo.realizedPnL) max_realized_pnl = posInfo.realizedPnL;
    double latest_dd = std::min(posInfo.inventoryPnL - max_unrealized_pnl, 0.0) + std::min(posInfo.realizedPnL - max_realized_pnl, 0.0);
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
    info["realized_pnl"] = posInfo.realizedPnL;
    info["leverage"] = posInfo.leverage;
    info["trade_count"] = static_cast<double>(tradeInfo.buy_trades + tradeInfo.sell_trades);
    info["buy_trades"] = static_cast<double>(tradeInfo.buy_trades);
    info["sell_trades"] = static_cast<double>(tradeInfo.sell_trades);
    info["buy_amount"] = tradeInfo.buy_amount;
    info["sell_amount"] = tradeInfo.sell_amount;
    info["drawdown"] = drawdown;
    info["fees"] = posInfo.fees;
    info["average_price"] = posInfo.averagePrice;
    info["net_position_usd"] = posInfo.netPosition;  // USD value of position
    info["net_amount_btc"] = strategy.getPosition().getNetAmount();  // BTC amount of position
    
    // Last placed quote prices for diagnostics (to verify actual spreads)
    info["last_bid_price"] = strategy.getLastBidPrice();
    info["last_ask_price"] = strategy.getLastAskPrice();
    info["last_mid_price"] = strategy.getLastMidPrice();
}


void EnvAdaptor::computeState(OrderBook& book)
{
    auto market_signals = market_builder->add_book(book);
    // Copy market signals [0..12] (13 signals)
    std::copy_n(market_signals.begin(), market_signals.size(), state.begin());
    
    // Compute AMM flow signals [13..15] (3 signals)
    double mid_price = (book.bid_prices[0] + book.ask_prices[0]) * 0.5;
    if (mid_price > 0) {
        AmmFlowSignals amm_signals = amm_simulator.step(mid_price);
        state[13] = amm_signals.net_flow;        // Cumulative flow direction
        state[14] = amm_signals.flow_imbalance;  // Recent buy/sell imbalance
        state[15] = amm_signals.inventory_delta; // LP inventory change
    }
    
    computeInfo(book);
}
