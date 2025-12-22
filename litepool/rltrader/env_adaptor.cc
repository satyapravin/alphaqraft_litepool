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

#include "env_adaptor.h"
#include <algorithm>
#include <iostream>

using namespace RLTrader;

EnvAdaptor::EnvAdaptor(Strategy& strat, BaseExchange& exch, const std::string& trade_filename, int ticks_per_step):
            strategy(strat),
            exchange(exch),
            ticks_per_step_(ticks_per_step),
            market_builder(std::make_unique<MarketSignalBuilder>()),
            trade_reader(trade_filename.empty() ? nullptr : 
                         std::make_unique<TradeReader>(trade_filename, 0)),
            trade_signal_builder(std::make_unique<TradeSignalBuilder>()),
            bid_prices(), ask_prices(), bid_sizes(), ask_sizes() {
}

bool EnvAdaptor::next() {
    std::fill_n(state.begin(), state.size(), 0);
    OrderBook book;
    size_t read_slot;
    
    // Advance multiple ticks per RL step to let orders persist
    for (int tick = 0; tick < ticks_per_step_; ++tick) {
        bool read_success = false;
        try {
            read_success = this->exchange.next_read(read_slot, book);
        } catch (const std::exception& e) {
            // Exception means no more data - return false to end episode
            return false;
        } catch (...) {
            return false;
        }
        
        // Guard: If no data available, episode must end immediately
        // Don't continue processing remaining ticks - return immediately
        if (!read_success) {
            return false;  // No more data - episode ends
        }
        
        try {
            this->strategy.next();  // Process any fills from this tick
        } catch (const std::exception& e) {
            this->exchange.done_read(read_slot);  // Still need to release the slot
            return false;
        } catch (...) {
            this->exchange.done_read(read_slot);
            return false;
        }
        
        this->exchange.done_read(read_slot);
    }
    
    // Compute state from the last tick
    // Guard: Only compute state if we successfully read data for all ticks
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
    
    // Reset trade signal builder
    if (trade_signal_builder) {
        trade_signal_builder->reset();
    }
    
    // Trade reader will be synced via syncTradeReader() call from Reset()
}

void EnvAdaptor::syncTradeReader(long long book_start_timestamp) {
    if (trade_reader) {
        trade_reader->reset(book_start_timestamp);
    }
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
    info["spread_capture"] = posInfo.spreadCapture;  // LIFO spread capture from closed round-trips
    info["leverage"] = posInfo.leverage;
    info["target_inventory"] = strategy.getTargetInventory();  // Agent's desired inventory level
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
    
    // Compute AMM flow signals [13..16] (4 signals)
    double mid_price = (book.bid_prices[0] + book.ask_prices[0]) * 0.5;
    if (mid_price > 0) {
        AmmFlowSignals amm_signals = amm_simulator.step(mid_price);
        state[13] = amm_signals.net_flow;        // EMA-based flow momentum
        state[14] = amm_signals.flow_imbalance;  // Recent buy/sell imbalance
        state[15] = amm_signals.inventory_delta; // LP inventory change
        
        // [16] Cumulative flow / balance: trend indicator for target inventory
        // Normalized by initial balance to give [-1, 1] scale for typical flow ranges
        double init_balance = strategy.getPosition().getInitialBalance();
        if (init_balance > 1e-9) {  // Guard against division by zero/near-zero
            // Scale: cumulative_flow of ±balance maps to ±1
            double flow_per_balance = amm_signals.cumulative_flow / init_balance;
            state[16] = std::tanh(flow_per_balance);  // Smooth bounding to [-1, 1]
        } else {
            state[16] = 0.0;
        }
    }
    
    // Trade signals [17..24] (8 signals)
    if (trade_reader && trade_signal_builder) {
        // Cast exchange to SimExchange to access getCurrentTimestamp()
        SimExchange* sim_exch = dynamic_cast<SimExchange*>(&exchange);
        if (sim_exch) {
            long long book_timestamp = sim_exch->getCurrentTimestamp();
            
            // Get trades up to current book timestamp (synchronized)
            std::vector<Trade> recent_trades = trade_reader->getRecentTrades(book_timestamp);
            
            // Compute trade signals (pass book timestamp for time_since_last_trade calculation)
            TradeSignals trade_signals = trade_signal_builder->add_trades(recent_trades, mid_price, book_timestamp);
            
            state[17] = trade_signals.buy_volume;
            state[18] = trade_signals.sell_volume;
            state[19] = trade_signals.volume_imbalance;
            state[20] = trade_signals.trade_intensity;
            state[21] = trade_signals.price_impact;
            state[22] = trade_signals.buy_pressure;
            state[23] = trade_signals.sell_pressure;
            state[24] = trade_signals.time_since_last_trade;
        } else {
            // Exchange doesn't support timestamp - zero out trade signals
            std::fill_n(state.begin() + 17, 8, 0.0);
        }
    } else {
        // No trade reader - zero out trade signals
        std::fill_n(state.begin() + 17, 8, 0.0);
    }
    
    // [25-29] Agent state: position, PnL, and performance metrics
    // Critical for agent to know its own state for inventory management and performance tracking
    auto posInfo = strategy.getPosition().getPositionInfo(book.bid_prices[0], book.ask_prices[0]);
    double initialBalance = strategy.getPosition().getInitialBalance();
    
    // [25] Current leverage (position / balance)
    state[25] = std::tanh(posInfo.leverage * 5.0);  // Scale: ±20% leverage maps to ±tanh(1) ≈ ±0.76
    
    // [26-29] Agent state signals - guard against division by zero/near-zero
    if (initialBalance > 1e-9) {
        // [26] Normalized position (netAmount normalized by initial balance)
        // For normal instruments: netAmount is in BTC, normalize by initialBalance to get position ratio
        double netAmount = strategy.getPosition().getNetAmount();
        double mid = 0.5 * (book.bid_prices[0] + book.ask_prices[0]);
        double positionValue = std::abs(netAmount * mid);
        state[26] = std::tanh((positionValue / initialBalance) * 2.0);  // Scale: 50% of balance = tanh(1) ≈ 0.76
        
        // [27] Normalized unrealized PnL (inventoryPnL / initialBalance)
        state[27] = std::tanh((posInfo.inventoryPnL / initialBalance) * 10.0);  // Scale: ±10% of balance = ±tanh(1)
        
        // [28] Normalized realized PnL (realizedPnL / initialBalance)
        state[28] = std::tanh((posInfo.realizedPnL / initialBalance) * 10.0);  // Scale: ±10% of balance = ±tanh(1)
        
        // [29] Normalized spread capture (spreadCapture / initialBalance)
        state[29] = std::tanh((posInfo.spreadCapture / initialBalance) * 10.0);  // Scale: ±10% of balance = ±tanh(1)
    } else {
        // Zero out agent state signals if initial balance is invalid
        state[26] = 0.0;
        state[27] = 0.0;
        state[28] = 0.0;
        state[29] = 0.0;
    }
    
    computeInfo(book);
}
