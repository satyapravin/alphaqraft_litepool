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
    this->exchange.fetchPosition(initQty, avgPrice, false);
    this->position.reset(initQty, avgPrice);
    this->order_id = 0;
}

void Strategy::updateTargetInventory(double target_inventory_action) {
    constexpr double TARGET_INVENTORY_SCALE = 1.0;
    double scaled_action = target_inventory_action * TARGET_INVENTORY_SCALE;
    
    target_inventory_ema = TARGET_EMA_ALPHA * scaled_action + 
                          (1.0 - TARGET_EMA_ALPHA) * target_inventory_ema;
    
    target_inventory_ema = std::clamp(target_inventory_ema, -TARGET_INVENTORY_SCALE, TARGET_INVENTORY_SCALE);
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
    // If long (leverage > 0): widen bid spread, tighten ask spread → more likely to sell
    // If short (leverage < 0): tighten bid spread, widen ask spread → more likely to buy
    // target_inventory_ema is already in [-0.02, +0.02] (target leverage range)
    double target_leverage = target_inventory_ema;  // Already in leverage units
    double inventory_error = leverage - target_leverage;
    
    // Skew factor: positive when long (widen bid, tighten ask), negative when short
    double skew_factor = -inventory_error * INVENTORY_SKEW_MULT;
    skew_factor = std::clamp(skew_factor, -1.5, 1.5);
    
    // Apply skew: widen one side, tighten the other
    // Positive skew_factor (long) → widen bid, tighten ask
    // Negative skew_factor (short) → tighten bid, widen ask
    // Use percentage of current spread (not base) so skew scales with wide spreads during volatility
    double avg_spread_bps = (bid_spread_bps + ask_spread_bps) * 0.5;
    double skew_adjustment_bps = skew_factor * avg_spread_bps * 0.3;  // Max 30% of average spread
    bid_spread_bps += skew_adjustment_bps;   // Widen bid when long (positive skew)
    ask_spread_bps -= skew_adjustment_bps;  // Tighten ask when long (positive skew)
    
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
    
    // Use 1% size per level for ladder quoting
    // With 5 levels, total exposure is 5% per side
    constexpr double SIZE_PER_LEVEL_PCT = 1.0;
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
    
    // Check position limits
    bool can_place_bids = (level_size >= minAmount) && (leverage < config.max_leverage);
    bool can_place_asks = (level_size >= minAmount) && (leverage > -config.max_leverage);
    
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
    for(const auto& order: fills) {
        position.onFill(order);
    }
    
    // Track steps since last fill (for observations)
    long current_trades = position.getNumberOfTrades();
    if (current_trades > prev_trade_count_) {
        // Fill happened - reset counter
        steps_since_last_fill_ = 0;
        prev_trade_count_ = current_trades;
    } else {
        // No fill - increment counter
        steps_since_last_fill_++;
    }
}
