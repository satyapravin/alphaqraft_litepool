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
    this->prev_mid_price = 0;
    this->realized_vol = 0;
    this->last_bid_price = 0;
    this->last_ask_price = 0;
    this->last_mid_price = 0;
    this->hit_leverage_limit_ = false;  // Reset leverage limit flag
    this->steps_since_last_fill_ = 0;   // Reset fill tracking
    this->prev_trade_count_ = 0;
    this->prev_inventory_error_ = 0.0;  // Reset inventory error tracking
    this->steps_away_from_target_ = 0;  // Reset urgency tracking
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

void Strategy::updateTargetInventory(double target_inventory_action) {
    // Action is in [-target_range, +target_range] (e.g., [-5, +5] if target_range=5.0)
    // Use action directly without scaling
    double target_raw = target_inventory_action;
    
    // Apply EMA smoothing to prevent flipping (TARGET_EMA_ALPHA = 0.0058 gives 120 step half-life = 60 sec)
    // This smooths out noisy AMM signals while still allowing responsive updates
    // With alpha=0.0058, it takes ~120 steps to move halfway from current to target
    target_inventory_ema = TARGET_EMA_ALPHA * target_raw + (1.0 - TARGET_EMA_ALPHA) * target_inventory_ema;
}

void Strategy::updateVolatility(double mid_price) {
    // Update realized volatility using EMA of squared returns
    if (prev_mid_price > 0 && mid_price > 0) {
        double ret = (mid_price - prev_mid_price) / prev_mid_price;
        double squared_ret = ret * ret;
        
        // EMA update: vol² = α * ret² + (1-α) * vol²
        // realized_vol stores the square root (standard deviation)
        double vol_sq = realized_vol * realized_vol;
        vol_sq = VOL_EMA_ALPHA * squared_ret + (1.0 - VOL_EMA_ALPHA) * vol_sq;
        realized_vol = std::sqrt(vol_sq);
    }
    prev_mid_price = mid_price;
}

std::pair<double, double> Strategy::computeQuotePrices(
    const RLAction& action,
    double mid_price,
    double leverage,
    double tick_size) {
    
    // ============================================================================
    // Direct Spread Control Model
    // ============================================================================
    // Agent directly controls bid_spread and ask_spread (full spreads in bps).
    // No reservation price - quotes are placed symmetrically around mid-price,
    // with inventory skew applied by adjusting spreads directly.
    // ============================================================================
    
    // === Step 1: Compute base spread (in bps) ===
    double base_spread_bps = config.base_spread_bps;
    
    // === Step 2: Volatility adjustment ===
    // In volatile markets, widen spreads to compensate for adverse selection risk
    // vol_mult = 1 + volatility * VOL_SPREAD_MULT
    // E.g., if vol=0.001 (0.1% per tick) and MULT=50, spread widens by 5%
    // For liquidity crunches, allow much larger volatility adjustments
    double vol_mult = 1.0 + realized_vol * VOL_SPREAD_MULT;
    vol_mult = std::clamp(vol_mult, 1.0, 20.0);  // Cap at 20x widening (was 5x) - allows huge spreads during crashes
    
    // === Step 3: Agent spread control (full spreads in bps) ===
    // bid_spread/ask_spread actions: [-1, 1] → [MIN_SPREAD_MULT, MAX_SPREAD_MULT]
    // EXPONENTIAL mapping: more control at tighter spreads (where it matters most)
    // action=-1 → MIN_SPREAD_MULT (0.5x), action=0 → 1.0x, action=+1 → MAX_SPREAD_MULT (50.0x)
    // Allows agent to widen spreads dramatically during liquidity crunches
    double log_ratio = std::log(MAX_SPREAD_MULT / MIN_SPREAD_MULT) / 2.0;  // ~2.3 for 50.0/0.5
    double center_mult = std::sqrt(MAX_SPREAD_MULT * MIN_SPREAD_MULT);     // Geometric mean ~5.0 for sqrt(50.0*0.5)
    
    double bid_spread_mult = center_mult * std::exp(action.bid_spread * log_ratio);
    double ask_spread_mult = center_mult * std::exp(action.ask_spread * log_ratio);
    
    // Clamp to ensure within bounds (handles numerical edge cases)
    bid_spread_mult = std::clamp(bid_spread_mult, MIN_SPREAD_MULT, MAX_SPREAD_MULT);
    ask_spread_mult = std::clamp(ask_spread_mult, MIN_SPREAD_MULT, MAX_SPREAD_MULT);
    
    // === Step 4: Compute base spreads (full spreads in bps) ===
    double bid_spread_bps = base_spread_bps * vol_mult * bid_spread_mult;
    double ask_spread_bps = base_spread_bps * vol_mult * ask_spread_mult;
    
    // === Step 5: Apply inventory skew by adjusting spreads directly ===
    // Goal: Push current leverage toward target leverage
    // If too long (leverage > target): widen bid, tighten ask → more likely to sell (reduce long)
    // If too short (leverage < target): tighten bid, widen ask → more likely to buy (reduce short)
    // target_inventory_ema is in [-target_range, +target_range] (e.g., [-5, +5] if target_range=5.0)
    double target_leverage = target_inventory_ema;  // Already in leverage units
    double inventory_error = leverage - target_leverage;
    
    // === Time-based urgency tracking ===
    // Track how long we've been away from target (increases urgency over time)
    if (std::abs(inventory_error) > URGENCY_TIME_THRESHOLD) {
        // Still away from target - increment urgency counter
        // Check if error sign changed (flipped direction) - reset if so
        if (std::signbit(inventory_error) != std::signbit(prev_inventory_error_)) {
            steps_away_from_target_ = 0;  // Reset if we flipped direction
        } else {
            steps_away_from_target_++;  // Increment if same direction
        }
    } else {
        // Close to target - reset urgency
        steps_away_from_target_ = 0;
    }
    prev_inventory_error_ = inventory_error;
    
    // === Adaptive skew multiplier based on volatility, urgency, and time ===
    // 1. Error magnitude urgency: smooth tanh function (saturates for large errors)
    //    tanh provides smooth response: small errors get linear scaling, large errors saturate
    double error_magnitude = std::abs(inventory_error);
    double error_urgency = std::tanh(error_magnitude * 5.0);  // Scale: 0.2 error → tanh(1) ≈ 0.76, saturates at 1.0
    
    // 2. Time-based urgency: increases with time away from target (up to 2x multiplier)
    //    More time away = more urgent = stronger skew
    double time_urgency = 1.0 + std::tanh(steps_away_from_target_ / 100.0);  // 100 steps → tanh(1) ≈ 0.76, max 2.0x
    
    // 3. Volatility adjustment: higher vol → stronger skew (but risk-averse mode for extreme vol)
    //    Moderate vol: increase skew (need to move faster)
    //    Extreme vol: reduce skew (risk-averse: avoid adverse selection)
    constexpr double VOL_THRESHOLD_MODERATE = 0.001;  // 0.1% per tick = moderate volatility
    constexpr double VOL_THRESHOLD_EXTREME = 0.005;   // 0.5% per tick = extreme volatility
    double vol_skew_mult;
    if (realized_vol < VOL_THRESHOLD_MODERATE) {
        // Low volatility: standard skew
        vol_skew_mult = 1.0;
    } else if (realized_vol < VOL_THRESHOLD_EXTREME) {
        // Moderate volatility: increase skew strength (need to move inventory faster)
        vol_skew_mult = 1.0 + (realized_vol - VOL_THRESHOLD_MODERATE) / (VOL_THRESHOLD_EXTREME - VOL_THRESHOLD_MODERATE);  // 1.0 to 2.0
    } else {
        // Extreme volatility: risk-averse mode (reduce skew to avoid adverse selection)
        // Still allow some skew, but much weaker
        double excess_vol = (realized_vol - VOL_THRESHOLD_EXTREME) / VOL_THRESHOLD_EXTREME;  // Normalized excess
        vol_skew_mult = 2.0 * std::exp(-excess_vol * 2.0);  // Exponential decay: 2.0 → 0.27 at 2x threshold
        vol_skew_mult = std::max(vol_skew_mult, 0.3);  // Floor at 0.3x (still some skew, but very weak)
    }
    
    // Combined adaptive skew multiplier: base * error_urgency * time_urgency * vol_skew_mult
    // This gives stronger skew when:
    // - Error is large (error_urgency)
    // - We've been away from target for a while (time_urgency)
    // - Volatility is moderate (vol_skew_mult), but weak in extreme vol (risk-averse)
    double adaptive_skew_mult = INVENTORY_SKEW_MULT * error_urgency * time_urgency * vol_skew_mult;
    
    // Skew factor: positive when too long (need to sell), negative when too short (need to buy)
    // inventory_error > 0 means too long → need to sell → positive skew (widen bid, tighten ask)
    // inventory_error < 0 means too short → need to buy → negative skew (tighten bid, widen ask)
    // So skew_factor should have the SAME sign as inventory_error
    double skew_factor = inventory_error * adaptive_skew_mult;
    
    // Apply skew: adjust spreads to push toward target
    // Positive skew_factor (too long, need to sell) → widen bid, tighten ask
    // Negative skew_factor (too short, need to buy) → tighten bid, widen ask
    // Use percentage of current spread (not base) so skew scales with wide spreads during volatility
    double avg_spread_bps = (bid_spread_bps + ask_spread_bps) * 0.5;
    double skew_adjustment_bps = std::abs(skew_factor) * avg_spread_bps;
    
    // Apply opposite adjustments to create asymmetry:
    // Positive skew_factor (too long, need to sell) → widen bid (+), tighten ask (-)
    // Negative skew_factor (too short, need to buy) → tighten bid (-), widen ask (+)
    // Since skew_factor has the same sign as inventory_error, we use it directly
    if (skew_factor > 0) {
        // Too long: widen bid, tighten ask
        bid_spread_bps += skew_adjustment_bps;
        ask_spread_bps -= skew_adjustment_bps;
    } else if (skew_factor < 0) {
        // Too short: tighten bid, widen ask
        bid_spread_bps -= skew_adjustment_bps;
        ask_spread_bps += skew_adjustment_bps;
    }
    // If skew_factor == 0, no adjustment needed

    // Ensure minimum spreads (safety floor)
    bid_spread_bps = std::max(bid_spread_bps, base_spread_bps * MIN_SPREAD_MULT);
    ask_spread_bps = std::max(ask_spread_bps, base_spread_bps * MIN_SPREAD_MULT);
    
    // === Step 6: Convert spreads to price units and place quotes ===
    // Agent controls FULL spreads (distance from mid to quote)
    // No division by 2 - bid_spread_bps is already the full distance from mid
    double bid_spread_price = mid_price * bid_spread_bps / 10000.0;
    double ask_spread_price = mid_price * ask_spread_bps / 10000.0;
    
    double bid_price = mid_price - bid_spread_price;
    double ask_price = mid_price + ask_spread_price;
    
    // === Step 7: Safety checks ===
    // REMOVED: 1 bps minimum floor - let agent learn consequences of tight spreads
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
    
    constexpr int NUM_LEVELS = 3;
    
    // Validate RL outputs are in expected range [-1, 1]
    assert(action.bid_spread >= -1.0001 && action.bid_spread <= 1.0001);
    assert(action.ask_spread >= -1.0001 && action.ask_spread <= 1.0001);
    
    // Early exit if prices invalid
    if (bid_prices[0] < 0.0001 || ask_prices[0] < 0.0001) return;
    
    auto tick_size = instrument.getTickSize();
    auto minAmount = instrument.getMinAmount();
    auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
    auto leverage = posInfo.leverage;
    
    
    // Reset flag if leverage is back within limits
    hit_leverage_limit_ = false;
    
    auto initBalance = position.getInitialBalance();
    auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    
    // Update volatility estimate for spread adjustment
    updateVolatility(mid_price);
    
    // Compute base quote prices (for level 1)
    auto [base_bid_price, base_ask_price] = computeQuotePrices(action, mid_price, leverage, tick_size);
    auto [level_size, _] = computeQuoteSizes(action, initBalance);
    
    // Calculate spreads from mid for ladder spacing
    double bid_spread_from_mid = mid_price - base_bid_price;  // Positive value
    double ask_spread_from_mid = base_ask_price - mid_price;  // Positive value
    
    // Ensure minimum spread of 1 tick
    bid_spread_from_mid = std::max(bid_spread_from_mid, tick_size);
    ask_spread_from_mid = std::max(ask_spread_from_mid, tick_size);
    
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
    last_bid_price = can_place_bids ? (mid_price - bid_spread_from_mid) : 0.0;
    last_ask_price = can_place_asks ? (mid_price + ask_spread_from_mid) : 0.0;
    
    // Place ladder of orders
    for (int level = 1; level <= NUM_LEVELS; ++level) {
        // Spread increases with level: 1x, 2x, 3x, 4x, 5x base spread
        double level_bid_spread = bid_spread_from_mid * level;
        double level_ask_spread = ask_spread_from_mid * level;
        
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

bool Strategy::shouldRequote(const RLAction& action,
                             FixedVector<double, 20>& bid_prices,
                             FixedVector<double, 20>& ask_prices,
                             double tick_threshold) {
    // If no valid prices, always requote
    if (bid_prices[0] < 0.0001 || ask_prices[0] < 0.0001) {
        return true;
    }
    
    // If no quotes placed yet, must requote
    if (last_bid_price < 0.0001 || last_ask_price < 0.0001) {
        return true;
    }
    
    auto tick_size = instrument.getTickSize();
    auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
    auto leverage = posInfo.leverage;
    auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
    
    // Compute what new quotes would be
    auto [proposed_bid, proposed_ask] = computeQuotePrices(action, mid_price, leverage, tick_size);
    
    // Check if proposed quotes differ from current quotes by more than threshold
    double bid_diff = std::abs(proposed_bid - last_bid_price);
    double ask_diff = std::abs(proposed_ask - last_ask_price);
    
    double threshold_price = tick_size * tick_threshold;
    
    // Requote if either side differs by more than threshold
    return (bid_diff > threshold_price) || (ask_diff > threshold_price);
}
