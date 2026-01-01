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

#include "strategy.h"
#include <algorithm>
#include <cassert>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <deque>
#include <vector>
#include "orderbook.h"

using namespace RLTrader;

Strategy::Strategy(BaseInstrument& instr, BaseExchange& exch, const double& balance, 
                   int maxTicks, const StrategyConfig& cfg)
    : instrument(instr), exchange(exch),
      position(instr, balance, 0, 0),
      config(cfg),
      order_id(0), max_ticks(maxTicks),
      target_inventory_ema(0) {
    assert(max_ticks >= 5);
}

void Strategy::reset() {
    // CRITICAL BUG FIX: Do NOT reset exchange here!
    // The exchange is already reset by RlTraderEnv::Reset() -> exchange_ptr->reset()
    // Resetting it again would cause CsvReader to be reset twice with different random start lines,
    // leading to expensive double sequential skips and state confusion.
    // this->exchange.reset();  // REMOVED - causes double reset bug
    
    double initQty = 0;
    double avgPrice = 0;
    this->target_inventory_ema = 0;
    this->risk_aversion_ema = 0.5;  // Reset to default [0, 1]
    this->vol_2min_ = 0.0;
    this->price_history_.clear();
    this->last_bid_price = 0;
    this->last_ask_price = 0;
    this->last_bid_action_ = 0.5;  // Reset spread action tracking
    this->last_ask_action_ = 0.5;
    this->last_mid_price = 0;
    this->hit_leverage_limit_ = false;  // Reset leverage limit flag
    this->steps_since_last_fill_ = 0;   // Reset fill tracking
    this->prev_trade_count_ = 0;
    this->last_netAmount_ = 0.0;         // Reset netAmount tracking
    this->last_trades_ = 0;              // Reset trade count tracking
    this->first_call_ = true;             // Reset first call flag
    
    // CRITICAL: fetchPosition() should return 0 for SimExchange
    // If it returns non-zero, that's the bug causing netAmount to be non-zero without trades
    this->exchange.fetchPosition(initQty, avgPrice, false);
    
    // CRITICAL FIX: Ensure initQty is 0 for SimExchange
    // If fetchPosition() returned non-zero, force it to 0 to prevent the bug
    // For SimExchange, position should always start at 0
    // For real exchanges, we might want to carry over position, but for now we'll force 0
    // TODO: Add a flag to distinguish SimExchange from real exchanges
    initQty = 0;
    avgPrice = 0;
    
    this->position.reset(initQty, avgPrice);
    this->order_id = 0;
    
    // Verify reset worked correctly
    double netAmount_after = this->position.getNetAmount();
    long trades_after = this->position.getNumberOfTrades();
    if (std::abs(netAmount_after) > 1e-9 || trades_after != 0) {
        // BUG: Position not properly reset!
        // This should never happen after reset()
    }
}

void Strategy::updateTargetInventory(double target_inventory_action, double risk_aversion_action) {
    // Target inventory: Action is in [-target_range, +target_range]
    // Use action directly without scaling
    double target_raw = target_inventory_action;
    
    // Apply EMA smoothing to prevent flipping (TARGET_EMA_ALPHA = 0.0058 gives 120 step half-life = 60 sec)
    target_inventory_ema = TARGET_EMA_ALPHA * target_raw + (1.0 - TARGET_EMA_ALPHA) * target_inventory_ema;
    
    // Risk aversion: Action is in [0, 1]
    // Clamp to valid range and apply EMA smoothing
    double risk_aversion_clamped = std::clamp(risk_aversion_action, 0.0, 1.0);
    risk_aversion_ema = TARGET_EMA_ALPHA * risk_aversion_clamped + (1.0 - TARGET_EMA_ALPHA) * risk_aversion_ema;
}

void Strategy::updateVolatility(double mid_price) {
    // Update 2-minute volatility using rolling window of price returns
    if (mid_price > 0) {
        // Add current price to history
        price_history_.push_back(mid_price);
        
        // Keep only last VOL_WINDOW_STEPS prices (2 minutes)
        if (price_history_.size() > VOL_WINDOW_STEPS) {
            price_history_.pop_front();
        }
        
        // Calculate volatility if we have enough data (need at least 2 prices for returns)
        if (price_history_.size() >= 2) {
            // Compute returns
            std::vector<double> returns;
            for (size_t i = 1; i < price_history_.size(); ++i) {
                double prev_price = price_history_[i - 1];
                double curr_price = price_history_[i];
                if (prev_price > 0) {
                    double ret = (curr_price - prev_price) / prev_price;
                    returns.push_back(ret);
                }
            }
            
            if (returns.size() >= 2) {
                // Calculate mean return
                double mean_ret = 0.0;
                for (double ret : returns) {
                    mean_ret += ret;
                }
                mean_ret /= returns.size();
                
                // Calculate variance
                double variance = 0.0;
                for (double ret : returns) {
                    double diff = ret - mean_ret;
                    variance += diff * diff;
                }
                variance /= returns.size();
                
                // Volatility is standard deviation of returns, multiplied by mid_price to get absolute volatility
                // This gives volatility in price units (same units as mid_price)
                vol_2min_ = std::sqrt(std::max(0.0, variance)) * mid_price;
            } else {
                vol_2min_ = 0.0;
            }
        } else {
            vol_2min_ = 0.0;
        }
    }
}

std::pair<double, double> Strategy::computeQuotePrices(
    const RLAction& action,
    double mid_price,
    double leverage,
    double tick_size) {
    
    // ============================================================================
    // Avellaneda-Stoikov Model Implementation
    // ============================================================================
    // Uses A-S formulas:
    //   Reservation price: r = s - (q - q_target) * γ * σ² * (T - t)
    //   Optimal spread: δ = (1/γ) * log(1 + γ/k) + (q - q_target) * γ * σ² * (T - t)
    // Where:
    //   s = mid_price, q = leverage (inventory), q_target = target_inventory_ema,
    //   (q - q_target) = inventory_error, γ = risk_aversion_ema,
    //   σ² = variance (vol_2min_²), T - t = time remaining, k = order flow intensity
    // ============================================================================
    
    // Get risk aversion parameter (γ) from inventory agent
    double gamma = risk_aversion_ema;
    
    // Compute variance (σ²) from 2-minute volatility
    double variance = vol_2min_ * vol_2min_;
    
    // Time remaining (T - t): use 1 hour as typical trading horizon
    // This represents how long we expect to hold the position
    constexpr double TIME_HORIZON_SEC = 1.0;  
        
    // Compute inventory error (difference between current leverage and target)
    // This is what drives the A-S adjustments, not absolute inventory
    double target_leverage = target_inventory_ema;
    double inventory_error = leverage - target_leverage; // Positive = too long, Negative = too short
    
    // Inventory adjustment: positive inventory_error (too long) → widen bid, tighten ask
    //                     negative inventory_error (too short) → tighten bid, widen ask
    // This helps the agent actually BUILD position when it wants to go long/short
    // Without aggressive skewing, fills are too balanced and position stays near 0
    double inventory_adjustment = inventory_error * gamma * variance / TIME_HORIZON_SEC;
    // Cap adjustment to reasonable range (2% of price max - increased from 1%)
    inventory_adjustment = std::clamp(inventory_adjustment, -mid_price * 0.02, mid_price * 0.02);
    
    // Agent controls bid_spread and ask_spread in [0, 1] range
    // These ADD spread on top of the base_spread_bps minimum
    // This prevents the agent from quoting too tight and getting adversely selected
    
    // Base spread from config (in basis points, convert to price units)
    // 1 bps = 0.01% = 0.0001, so base_spread_bps * mid_price / 10000
    double base_spread = mid_price * config.base_spread_bps / 10000.0;
    
    // Minimum spread: max of (base_spread, tick_size)
    // This ensures even with action=0, we have meaningful spread to avoid adverse selection
    double min_spread = std::max(base_spread, tick_size);
    
    // Agent action [0, 1] ADDS more spread on top of minimum
    // action=0 → minimum spread (base_spread_bps) - tightest possible
    // action=1 → minimum + volatility extra - widest
    double vol_extra = std::max(vol_2min_, tick_size);  // Extra spread range based on volatility
    double bid_action = min_spread + std::clamp(action.bid_spread, 0.0, 1.0) * vol_extra;
    double ask_action = min_spread + std::clamp(action.ask_spread, 0.0, 1.0) * vol_extra;
    
    double bid_spread = bid_action + inventory_adjustment;  // Widen bid when positive error
    double ask_spread = ask_action - inventory_adjustment;  // Tighten ask when positive error
    
    // Asymmetric floor: only protect the position-INCREASING side
    // Allow the position-REDUCING side to go tight for faster offloading
    double net_position = leverage;  // Positive = long, negative = short
    
    if (net_position > 0) {
        // Agent is LONG: protect bid (would increase long), allow tight ask (reduces long)
        bid_spread = std::max(bid_spread, min_spread);
        ask_spread = std::max(ask_spread, tick_size);  // Minimum 1 tick
    } else if (net_position < 0) {
        // Agent is SHORT: protect ask (would increase short), allow tight bid (reduces short)
        bid_spread = std::max(bid_spread, tick_size);  // Minimum 1 tick
        ask_spread = std::max(ask_spread, min_spread);
    } else {
        // Flat: protect both sides equally
        bid_spread = std::max(bid_spread, min_spread);
        ask_spread = std::max(ask_spread, min_spread);
    }
    
    // Quotes are placed around mid price
    double bid_price = mid_price - bid_spread;
    double ask_price = mid_price + ask_spread;
   
    // Agent will experience adverse selection if quoting too tight, which is the learning signal
    
    // Only ensure minimum spread of 1 tick between bid and ask (hard floor)
    if (ask_price - bid_price < tick_size) {
        double center = (bid_price + ask_price) / 2.0;
        bid_price = center - tick_size / 2.0;
        ask_price = center + tick_size / 2.0;
    }
    
    // Round to tick size (bid down, ask up to maintain spread)
    bid_price = std::floor(bid_price / tick_size) * tick_size;
    ask_price = std::ceil(ask_price / tick_size) * tick_size;
    
    return {bid_price, ask_price};
}

std::pair<double, double> Strategy::computeQuoteSizes(
    const RLAction& action,
    double init_balance) {
    
    constexpr double SIZE_PER_LEVEL_PCT = 1;
    double size_usd = SIZE_PER_LEVEL_PCT / 100.0 * init_balance;
    
    return {size_usd, size_usd};
}

void Strategy::quote(const RLAction& action,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
    // ============================================================================
    // LADDER QUOTING: Place 5 levels of orders per side
    // ============================================================================
    // Level 1: 1x base spread (closest to mid, fills first)
    // Level 2: 2x base spread
    // Level 3: 3x base spread
    // Level 4: 4x base spread
    // Level 5: 5x base spread (furthest from mid, fills last)
    //
    // Benefits:
    // - Reduced adverse selection (only closest level fills on small moves)
    // - Natural dollar-cost averaging on big moves
    // - More rebates (5 fills = 5x rebates)
    // - More realistic market making
    // ============================================================================
    
    constexpr int NUM_LEVELS = 5;
    
    // Validate RL outputs are in expected range [0, 1]
    assert(action.bid_spread >= -0.0001 && action.bid_spread <= 1.0001);
    assert(action.ask_spread >= -0.0001 && action.ask_spread <= 1.0001);
    
    // Store spread actions for tracking
    last_bid_action_ = std::clamp(action.bid_spread, 0.0, 1.0);
    last_ask_action_ = std::clamp(action.ask_spread, 0.0, 1.0);
    
    // Early exit if prices invalid
    if (bid_prices[0] < 0.0001 || ask_prices[0] < 0.0001) return;
    
    // Force requote conditions:
    // 1. First quote (no orders placed yet)
    // 2. No active orders in book (they got filled - need to replace them)
    // 3. Quotes are stale (too far from current mid - prevents stuck quotes)
    bool first_quote = (last_bid_price < 0.0001 && last_ask_price < 0.0001);
    bool no_active_orders = exchange.getBidOrders().empty() && exchange.getAskOrders().empty();
    
    // Stale quote detection: if quotes are too far from mid, force requote
    // This prevents the agent from sitting on stale quotes that never fill
    auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
    bool stale_quotes = false;
    if (last_bid_price > 0.0001 && last_ask_price > 0.0001 && mid_price > 0.0001) {
        double bid_distance_bps = (mid_price - last_bid_price) / mid_price * 10000.0;
        double ask_distance_bps = (last_ask_price - mid_price) / mid_price * 10000.0;
        // If either quote is more than 5 bps from mid, force requote
        stale_quotes = (bid_distance_bps > 5.0) || (ask_distance_bps > 5.0);
    }
    
    bool force_requote = first_quote || no_active_orders || stale_quotes;
    
    // Agent controls requote decision: >0.3 = requote, <=0.3 = keep existing orders
    // This gives the agent explicit control over quote update timing
    // Skip quoting if agent says no requote (keep existing orders in the book)
    // BUT always requote if forced (first call, no orders, or stale quotes)
    if (!force_requote && action.should_requote <= 0.3) {
        // Still update volatility even when not requoting
        updateVolatility(mid_price);
        return;
    }
    
    auto tick_size = instrument.getTickSize();
    auto minAmount = instrument.getMinAmount();
    auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
    auto leverage = posInfo.leverage;
    
    
    // Reset flag if leverage is back within limits
    hit_leverage_limit_ = false;
    
    auto initBalance = position.getInitialBalance();
    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    
    // Update volatility estimate for spread adjustment
    updateVolatility(mid_price);
    
    auto [base_bid_price, base_ask_price] = computeQuotePrices(action, mid_price, leverage, tick_size);
    auto [level_size, _] = computeQuoteSizes(action, initBalance);
    
    double bid_spread_from_res = mid_price - base_bid_price;  // Positive value
    double ask_spread_from_res = base_ask_price - mid_price;  // Positive value
    
    // Ensure minimum spread of 1 tick
    bid_spread_from_res = std::max(bid_spread_from_res, tick_size);
    ask_spread_from_res = std::max(ask_spread_from_res, tick_size);
    
    // Cancel existing orders before placing new ones
    exchange.cancelOrders();
    
    // Convert size to trade amount
    level_size = instrument.getTradeAmount(level_size, mid_price);
    
    // Check position limits - prevent placing orders that would push leverage beyond limits
    // Use stricter check: only place if current leverage is well within limits (95% threshold)
    // This prevents fills from pushing leverage too far beyond the limit
    bool can_place_bids = (level_size >= minAmount) && (leverage < config.max_leverage * 0.95);
    bool can_place_asks = (level_size >= minAmount) && (leverage > -config.max_leverage * 0.95);
    
    // Store first level prices for diagnostics
    last_mid_price = mid_price;
    last_bid_price = can_place_bids ? base_bid_price : 0.0;
    last_ask_price = can_place_asks ? base_ask_price : 0.0;
    
    for (int level = 1; level <= NUM_LEVELS; ++level) {
        double level_bid_spread = bid_spread_from_res * level;
        double level_ask_spread = ask_spread_from_res * level;
        
        double bid_price = mid_price - level_bid_spread;
        double ask_price = mid_price + level_ask_spread;


        // Round to tick size
        bid_price = std::floor(bid_price / tick_size) * tick_size;
        ask_price = std::ceil(ask_price / tick_size) * tick_size;
        
        // Prevent crossing
        if (bid_price >= best_ask) {
            bid_price = best_ask - tick_size * level;
        }
        if (ask_price <= best_bid) {
            ask_price = best_bid + tick_size * level;
        }
        
        // Place bid at this level
        if (can_place_bids && bid_price > 0) {
            this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, level_size);
        }
        
        // Place ask at this level
        if (can_place_asks && ask_price > 0) {
            this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, level_size);
        }

        if (leverage >= 1.0 || leverage <= -1.0) {
            hit_leverage_limit_ = true;  // Flag that leverage limit was hit
        }    
    }
}

void Strategy::next() {
    auto fills = exchange.getFills();
    
    // Track state before processing fills (ALWAYS, not just when fills exist)
    long trades_before = position.getNumberOfTrades();
    double netAmount_before = position.getNetAmount();
    auto trade_info_before = position.getTradeInfo();
    
    // Process fills
    for(const auto& order: fills) {
        // Verify order state before processing
        if (order.state != OrderState::FILLED) {
            // This should never happen - getFills() should only return FILLED orders
            continue;  // Skip non-filled orders
        }
        
        position.onFill(order);
    }
    
    // ALWAYS check if netAmount changed without trades incrementing (not just when fills exist)
    long trades_after = position.getNumberOfTrades();
    double netAmount_after = position.getNetAmount();
    auto trade_info_after = position.getTradeInfo();
    
    if (std::abs(netAmount_after - netAmount_before) > 1e-9 && trades_after == trades_before) {
        // BUG DETECTED: netAmount changed without trades incrementing!
        // This should never happen - onFill() always increments numOfTrades
        // 
        // Check if trade_info changed
        long buy_trades_before = trade_info_before.buy_trades;
        long sell_trades_before = trade_info_before.sell_trades;
        long buy_trades_after = trade_info_after.buy_trades;
        long sell_trades_after = trade_info_after.sell_trades;
        
        if (buy_trades_after == buy_trades_before && sell_trades_after == sell_trades_before) {
            // trade_info didn't change either - this is the bug!
            // onFill() was called but didn't update trade_info or numOfTrades
            // OR netAmount is being modified outside of onFill()
            throw std::runtime_error(
                "BUG: netAmount changed from " + std::to_string(netAmount_before) + 
                " to " + std::to_string(netAmount_after) + 
                " but trades didn't increment! (trades=" + std::to_string(trades_after) + 
                ", fills=" + std::to_string(fills.size()) + 
                ", buy_trades=" + std::to_string(buy_trades_after) + 
                ", sell_trades=" + std::to_string(sell_trades_after) + ")"
            );
        }
    }
    
    // Also check if netAmount changed between calls (before processing fills)
    // This would indicate netAmount is being modified elsewhere
    // NOTE: Using instance variables (not static) so each Strategy instance tracks its own state
    // Compare netAmount_before (start of this call) to last_netAmount_ (end of previous call)
    if (!first_call_) {
        if (std::abs(netAmount_before - last_netAmount_) > 1e-9 && trades_before == last_trades_ && fills.empty()) {
            // BUG: netAmount changed between calls without any fills!
            // This means netAmount is being modified outside of onFill() or reset()
            throw std::runtime_error(
                "BUG: netAmount changed from " + std::to_string(last_netAmount_) + 
                " to " + std::to_string(netAmount_before) + 
                " between calls without any fills! (trades=" + std::to_string(trades_before) + 
                ", netAmount_diff=" + std::to_string(netAmount_before - last_netAmount_) + ")"
            );
        }
    } else {
        first_call_ = false;
    }
    
    // Update tracking variables with values AFTER processing fills
    last_netAmount_ = netAmount_after;
    last_trades_ = trades_after;
    
    // Track steps since last fill (for observations)
    auto current_trades = position.getNumberOfTrades();
    if (current_trades > prev_trade_count_) {
        // Fill happened - reset counter
        steps_since_last_fill_ = 0;
        prev_trade_count_ = current_trades;
    } else {
        // No fill - increment counter
        steps_since_last_fill_++;
    }
}

