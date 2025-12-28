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
    // IMPORTANT: Process AMM for EACH tick to capture all price movements
    for (int tick = 0; tick < ticks_per_step_; ++tick) {
        bool read_success = this->exchange.next_read(read_slot, book);
        
        // Guard: If no data available, episode must end immediately
        // Don't continue processing remaining ticks - return immediately
        if (!read_success) {
            return false;  // No more data - episode ends
        }
        
        this->strategy.next();  // Process any fills from this tick
        
        // Process ALL signal builders for EACH tick to capture full temporal dynamics
        // This ensures we don't lose 80% of information (5 ticks per RL step at 100ms each)
        double mid_price = (book.bid_prices[0] + book.ask_prices[0]) * 0.5;
        
        // 1. AMM flow signals - captures price movements for inventory simulation
        if (mid_price > 0) {
            amm_simulator.step(mid_price);
        }
        
        // 2. Market signals - maintains 5-snapshot window for spread/depth dynamics
        // Store last tick's signals for use in computeState()
        last_market_signals_ = market_builder->add_book(book);
        
        // 3. Trade signals - update EMA with trades from this tick's timestamp
        if (trade_reader && trade_signal_builder) {
            SimExchange* sim_exch = dynamic_cast<SimExchange*>(&exchange);
            if (sim_exch) {
                long long book_timestamp = sim_exch->getCurrentTimestamp();
                std::vector<Trade> recent_trades = trade_reader->getRecentTrades(book_timestamp);
                last_trade_signals_ = trade_signal_builder->add_trades(recent_trades, mid_price, book_timestamp);
            }
        }
        
        this->exchange.done_read(read_slot);
    }
    
    // Compute final state from the last tick
    // Signal builders already accumulated all ticks above, just read final values
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

bool EnvAdaptor::shouldRequote(const RLAction& action, double tick_threshold) {
    return this->strategy.shouldRequote(action, bid_prices, ask_prices, tick_threshold);
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
    prev_mid_price_ = 0.0;  // Reset previous mid price
    // Reset AMM simulator so it auto-initializes on first step with valid price
    amm_simulator.clear();
    
    // Reset EMA smoothing for AMM signals
    flow_imbalance_ema_ = 0.0;
    inventory_delta_ema_ = 0.0;
    
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
    info["initial_balance"] = strategy.getPosition().getInitialBalance();  // For average cost realized PnL calculation
    info["unrealized_pnl"] = posInfo.inventoryPnL;           // Weighted-average unrealized PnL (for logging, matches balance cash flow)
    info["lifo_unrealized_pnl"] = posInfo.lifoUnrealizedPnL; // LIFO unrealized (for rewards, consistent with spread_capture)
    info["realized_pnl"] = posInfo.realizedPnL;  // Weighted-average realized PnL (for logging, matches balance cash flow)
    info["spread_capture"] = posInfo.spreadCapture;  // LIFO spread capture from closed round-trips (for rewards)
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
    
    info["market_bid_price"] = bid_price;
    info["market_ask_price"] = ask_price;
    
    info["net_amount_btc_raw"] = strategy.getPosition().getNetAmount();
    info["average_price_raw"] = posInfo.averagePrice;
}


void EnvAdaptor::computeState(OrderBook& book)
{
    // Use cached market signals from last tick in next() loop
    // (market_builder was already called for each tick to maintain full temporal dynamics)
    if (!last_market_signals_.empty()) {
        std::copy_n(last_market_signals_.begin(), 
                    std::min(last_market_signals_.size(), static_cast<size_t>(13)), 
                    state.begin());
    }
    
    // Get AMM flow signals [13..16] (4 signals)
    // Note: AMM was already stepped for each tick in next() loop - just read final state
    if (amm_simulator.isInitialized()) {
        AmmFlowSignals amm_signals = amm_simulator.getSignals();
        state[13] = amm_signals.net_flow;        // EMA-based flow momentum (already smooth)
        
        // Apply EMA smoothing to noisy signals before storing in state
        flow_imbalance_ema_ = AMM_SIGNAL_EMA_ALPHA * amm_signals.flow_imbalance + 
                             (1.0 - AMM_SIGNAL_EMA_ALPHA) * flow_imbalance_ema_;
        state[14] = flow_imbalance_ema_;  // Smoothed buy/sell imbalance
        
        inventory_delta_ema_ = AMM_SIGNAL_EMA_ALPHA * amm_signals.inventory_delta + 
                              (1.0 - AMM_SIGNAL_EMA_ALPHA) * inventory_delta_ema_;
        state[15] = inventory_delta_ema_;  // Smoothed LP inventory change
        
        // [16] Cumulative flow / balance: trend indicator for target inventory
        // Normalized by initial balance to give [-1, 1] scale for typical flow ranges
        double init_balance = strategy.getPosition().getInitialBalance();
        if (init_balance > 1e-9) {  // Guard against division by zero/near-zero
            // Scale: cumulative_flow of ±balance maps to ±1
            double flow_per_balance = amm_signals.cumulative_flow / init_balance;
            state[16] = std::tanh(flow_per_balance);  // Smooth bounding to [-1, 1] (already smooth)
        } else {
            state[16] = 0.0;
        }
    }
    
    // Trade signals [17..24] (8 signals)
    // Use cached trade signals from last tick in next() loop
    if (trade_reader && trade_signal_builder) {
        state[17] = last_trade_signals_.buy_volume;
        state[18] = last_trade_signals_.sell_volume;
        state[19] = last_trade_signals_.volume_imbalance;
        state[20] = last_trade_signals_.trade_intensity;
        state[21] = last_trade_signals_.price_impact;
        state[22] = last_trade_signals_.buy_pressure;
        state[23] = last_trade_signals_.sell_pressure;
        state[24] = last_trade_signals_.time_since_last_trade;
    } else {
        // No trade reader - zero out trade signals
        std::fill_n(state.begin() + 17, 8, 0.0);
    }
    
    // [25-35] Agent state (11 signals): leverage, position, P&L, deviation, unrealized P/L %, 
    //         target inventory, entry price distance, time since fill, quote distance
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
        
        // [30] Deviation from target leverage (leverage - target_leverage)
        // Positive when over-leveraged, negative when under-leveraged
        // Normalized using tanh to bound to [-1, 1] range
        double target_leverage = strategy.getTargetInventory();
        double leverage_deviation = posInfo.leverage - target_leverage;
        state[30] = std::tanh(leverage_deviation * 10.0);  // Scale: ±0.1 deviation maps to ±tanh(1) ≈ ±0.76
        
        // [31] Unrealized P/L as % of position value
        // Tells the agent "I'm up/down X% on my current position"
        // Agent can learn to close at ±1% threshold
        if (positionValue > 1e-9) {  // Guard against division by zero when no position
            double unrealized_pnl_pct = posInfo.inventoryPnL / positionValue;
            state[31] = std::tanh(unrealized_pnl_pct * 100.0);  // Scale: ±1% maps to ±tanh(1) ≈ ±0.76
        } else {
            state[31] = 0.0;  // No position, no % P/L
        }
        
        // [32] Target inventory EMA - agent's smoothed target inventory
        // Inv agent can see what target it asked for vs current leverage
        state[32] = std::tanh(strategy.getTargetInventory() * 10.0);  // Scale: ±0.1 maps to ±tanh(1)
        
        // [33] Entry price distance - how far avg entry is from current mid (in %)
        // Positive = underwater (bought above current price), Negative = in profit
        if (posInfo.averagePrice > 1e-9 && mid > 1e-9) {
            double entry_distance_pct = (posInfo.averagePrice - mid) / mid;
            // Adjust sign based on position direction
            double netAmount = strategy.getPosition().getNetAmount();
            if (netAmount < 0) {
                // Short position: higher entry = profit, so flip sign
                entry_distance_pct = -entry_distance_pct;
            }
            state[33] = std::tanh(entry_distance_pct * 100.0);  // Scale: ±1% maps to ±tanh(1)
        } else {
            state[33] = 0.0;  // No position or invalid price
        }
        
        // [34] Time since last fill - normalized steps since last trade
        // Helps MM agent understand fill rate
        int steps_since_fill = strategy.getStepsSinceLastFill();
        state[34] = std::tanh(steps_since_fill / 100.0);  // Scale: 100 steps maps to tanh(1) ≈ 0.76
        
        // [35] Quote mid distance - how far our quotes are from mid (avg of bid/ask spread in bps)
        // Helps MM agent understand quote positioning
        double last_bid = strategy.getLastBidPrice();
        double last_ask = strategy.getLastAskPrice();
        if (mid > 1e-9 && last_bid > 0 && last_ask > 0) {
            double bid_distance_bps = (mid - last_bid) / mid * 10000.0;
            double ask_distance_bps = (last_ask - mid) / mid * 10000.0;
            double avg_distance_bps = (bid_distance_bps + ask_distance_bps) / 2.0;
            state[35] = std::tanh(avg_distance_bps / 10.0);  // Scale: 10 bps maps to tanh(1) ≈ 0.76
            
            // [36] Previous actual bid/ask spread (ask - bid, normalized in bps)
            // Helps MM agent understand what it actually quoted and compare with market spread
            double actual_spread_bps = (last_ask - last_bid) / mid * 10000.0;
            state[36] = std::tanh(actual_spread_bps / 20.0);  // Scale: 20 bps maps to tanh(1) ≈ 0.76
            
            // [37] Bid distance from mid (mid - last_bid, normalized in bps)
            // Helps MM agent understand bid quote positioning separately
            state[37] = std::tanh(bid_distance_bps / 10.0);  // Scale: 10 bps maps to tanh(1) ≈ 0.76
            
            // [38] Ask distance from mid (last_ask - mid, normalized in bps)
            // Helps MM agent understand ask quote positioning separately
            state[38] = std::tanh(ask_distance_bps / 10.0);  // Scale: 10 bps maps to tanh(1) ≈ 0.76
        } else {
            state[35] = 0.0;  // No quotes yet
            state[36] = 0.0;  // No quotes yet
            state[37] = 0.0;  // No quotes yet
            state[38] = 0.0;  // No quotes yet
        }
    } else {
        // Zero out agent state signals if initial balance is invalid
        state[26] = 0.0;
        state[27] = 0.0;
        state[28] = 0.0;
        state[29] = 0.0;
        state[30] = 0.0;
        state[31] = 0.0;
        state[32] = 0.0;
        state[33] = 0.0;
        state[34] = 0.0;
        state[35] = 0.0;
        state[36] = 0.0;
        state[37] = 0.0;
        state[38] = 0.0;
    }
    
    // [39] Mid price change - normalized price return (for inventory agent to understand price direction)
    double mid = 0.5 * (book.bid_prices[0] + book.ask_prices[0]);
    if (mid > 1e-9 && prev_mid_price_ > 1e-9) {
        double mid_change_pct = (mid - prev_mid_price_) / prev_mid_price_;
        state[39] = std::tanh(mid_change_pct * 1000.0);  // Scale: ±0.1% maps to ±tanh(1) ≈ ±0.76
    } else {
        state[39] = 0.0;  // No previous mid price
    }
    prev_mid_price_ = mid;  // Update for next step
    
    computeInfo(book);
}
