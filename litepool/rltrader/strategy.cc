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
    this->exchange.reset();
    double initQty = 0;
    double avgPrice = 0;
    this->target_inventory_ema = 0;
    this->prev_mid_price = 0;
    this->realized_vol = 0;
    this->last_bid_price = 0;
    this->last_ask_price = 0;
    this->last_mid_price = 0;
    this->exchange.fetchPosition(initQty, avgPrice, false);
    this->position.reset(initQty, avgPrice);
    this->order_id = 0;
}

void Strategy::updateTargetInventory(double target_inventory_action) {
    // RL action is already in [-1, 1], representing target leverage from -100% to +100%
    // No scaling - let agent express full range of inventory preferences
    // The actual position is still hard-limited to ±10%, but agent can "desire" more
    // This gives clearer signal about directional preference
    
    // EMA smoothing: target_inventory_ema = alpha * new_target + (1 - alpha) * old_target
    // TARGET_EMA_ALPHA = 0.05 means ~20 step half-life (smooth updates)
    target_inventory_ema = TARGET_EMA_ALPHA * target_inventory_action + 
                          (1.0 - TARGET_EMA_ALPHA) * target_inventory_ema;
    
    // Clamp to [-1, 1] - full range allowed
    target_inventory_ema = std::clamp(target_inventory_ema, -1.0, 1.0);
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
    // Avellaneda-Stoikov Inspired Market Making Model
    // ============================================================================
    // Key insight: A market maker controls two things:
    //   1. WHERE to quote (reservation price = adjusted fair value)
    //   2. HOW WIDE to quote (spread = risk/reward tradeoff)
    //
    // The model separates these concerns cleanly:
    //   - Reservation price is shifted based on inventory (reduces adverse selection)
    //   - Spread is widened based on volatility (compensates for risk)
    //   - RL agent fine-tunes both through actions
    // ============================================================================
    
    // === Step 1: Compute base half-spread ===
    double base_half_spread = mid_price * config.base_spread_bps / 10000.0 / 2.0;
    
    // === Step 2: Volatility adjustment ===
    // In volatile markets, widen spreads to compensate for adverse selection risk
    // vol_mult = 1 + volatility * VOL_SPREAD_MULT
    // E.g., if vol=0.001 (0.1% per tick) and MULT=50, spread widens by 5%
    double vol_mult = 1.0 + realized_vol * VOL_SPREAD_MULT;
    vol_mult = std::clamp(vol_mult, 1.0, 5.0);  // Cap at 5x widening
    
    // === Step 3: Agent spread control ===
    // bid_spread/ask_spread actions: [-1, 1] → [MIN_SPREAD_MULT, MAX_SPREAD_MULT]
    // EXPONENTIAL mapping: more control at tighter spreads (where it matters most)
    // action=-1 → MIN_SPREAD_MULT (0.2x), action=0 → 1.0x, action=+1 → MAX_SPREAD_MULT (3.0x)
    // Formula: mult = exp(action * log_ratio) where log_ratio = ln(MAX/MIN)/2
    double log_ratio = std::log(MAX_SPREAD_MULT / MIN_SPREAD_MULT) / 2.0;  // ~1.35
    double center_mult = std::sqrt(MAX_SPREAD_MULT * MIN_SPREAD_MULT);     // Geometric mean ~0.77
    
    double bid_spread_mult = center_mult * std::exp(action.bid_spread * log_ratio);
    double ask_spread_mult = center_mult * std::exp(action.ask_spread * log_ratio);
    
    // Clamp to ensure within bounds (handles numerical edge cases)
    bid_spread_mult = std::clamp(bid_spread_mult, MIN_SPREAD_MULT, MAX_SPREAD_MULT);
    ask_spread_mult = std::clamp(ask_spread_mult, MIN_SPREAD_MULT, MAX_SPREAD_MULT);
    
    // === Step 4: Compute final half-spreads ===
    double bid_half_spread = base_half_spread * vol_mult * bid_spread_mult;
    double ask_half_spread = base_half_spread * vol_mult * ask_spread_mult;
    
    // === Step 5: Compute reservation price (adjusted fair value) ===
    // The key insight from Avellaneda-Stoikov: when holding inventory,
    // shift your quotes to incentivize trades that reduce inventory.
    //
    // If long (leverage > 0): shift quotes DOWN (lower reservation price)
    //   → bid becomes less attractive (lower), ask becomes more attractive (lower)
    //   → more likely to sell, reduce long position
    //
    // If short (leverage < 0): shift quotes UP (higher reservation price)
    //   → bid becomes more attractive (higher), ask becomes less attractive (higher)
    //   → more likely to buy, reduce short position
    
    // Inventory error: how far we are from target
    // target_inventory_ema is in [-1, 1], scale to leverage units [-0.1, 0.1]
    double target_leverage = target_inventory_ema * 0.1;  // Scale to match leverage range
    double inventory_error = leverage - target_leverage;
    
    // Inventory-based skew: shift mid-point to reduce inventory error
    // Negative sign: positive error (too long) → negative shift (quote lower)
    // SIMPLIFIED: Removed action.skew - target_inventory is the only inventory control
    // This avoids conflicts where agent could set opposing target_inventory and skew
    double total_skew = -inventory_error * INVENTORY_SKEW_MULT;
    total_skew = std::clamp(total_skew, -1.5, 1.5);
    
    // Compute reservation price shift (in price units)
    double skew_shift = total_skew * base_half_spread;
    double reservation_price = mid_price + skew_shift;
    
    // === Step 6: Compute final quote prices ===
    // IMPORTANT: Ensure spreads are wide enough to keep quotes on correct side of mid
    // If skew shifts reservation UP, bid needs extra spread to stay below mid
    // If skew shifts reservation DOWN, ask needs extra spread to stay above mid
    double min_bid_half_spread = std::max(bid_half_spread, skew_shift + base_half_spread * 0.5);
    double min_ask_half_spread = std::max(ask_half_spread, -skew_shift + base_half_spread * 0.5);
    
    double bid_price = reservation_price - min_bid_half_spread;
    double ask_price = reservation_price + min_ask_half_spread;
    
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
    
    // Use fixed size (min_size_pct) - no RL control over size
    // This simplifies the action space and focuses learning on spread/skew
    double size_pct = config.min_size_pct;
    double size_usd = size_pct / 100.0 * init_balance;
    
    return {size_usd, size_usd};
}

void Strategy::quote(const RLAction& action,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
    // Validate RL outputs are in expected range [-1, 1]
    assert(action.bid_spread >= -1.0001 && action.bid_spread <= 1.0001);
    assert(action.ask_spread >= -1.0001 && action.ask_spread <= 1.0001);
    // Note: skew action removed - now computed automatically from inventory error
    
    // Early exit if prices invalid
    if (bid_prices[0] < 0.0001 || ask_prices[0] < 0.0001) return;
    
    auto tick_size = instrument.getTickSize();
    auto minAmount = instrument.getMinAmount();
    auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
    auto leverage = posInfo.leverage;
    auto initBalance = position.getInitialBalance();
    auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
    
    // Update volatility estimate for spread adjustment
    updateVolatility(mid_price);
    
    // Compute quote prices and sizes using Avellaneda-Stoikov inspired model
    auto [bid_price, ask_price] = computeQuotePrices(action, mid_price, leverage, tick_size);
    auto [bid_size_0, ask_size_0] = computeQuoteSizes(action, initBalance);
    
    // Ensure quotes don't cross the book
    // IMPORTANT: Don't clamp quotes to market edges - respect the computed spread
    // If computed spread is wide, quotes should be far from market (won't fill easily)
    // If computed spread is tight, quotes will be competitive (will fill more easily)
    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    
    // Only prevent crossing - don't clamp to market edges
    // This allows the agent to control how competitive its quotes are
    if (bid_price >= best_ask) {
        // Computed bid would cross - adjust to just below best ask
        bid_price = best_ask - tick_size;
    }
    if (ask_price <= best_bid) {
        // Computed ask would cross - adjust to just above best bid
        ask_price = best_bid + tick_size;
    }
    
    // CRITICAL: After crossing prevention, ensure quotes are still on correct side of mid
    // Use tick_size as minimum (not 1 bps) - let agent learn from tight spreads
    if (bid_price > mid_price - tick_size) {
        bid_price = mid_price - tick_size;
        bid_price = std::floor(bid_price / tick_size) * tick_size;
    }
    if (ask_price < mid_price + tick_size) {
        ask_price = mid_price + tick_size;
        ask_price = std::ceil(ask_price / tick_size) * tick_size;
    }
    
    // Ensure valid spread (bid < ask)
    if (bid_price >= ask_price) {
        // Invalid spread - use minimum 1-tick spread centered at mid
        double center = (best_bid + best_ask) * 0.5;
        bid_price = center - tick_size / 2.0;
        ask_price = center + tick_size / 2.0;
    }
    
    // Cancel existing orders before placing new ones
    exchange.cancelOrders();
    
    // Convert to trade amounts (rounded to minAmount)
    bid_size_0 = instrument.getTradeAmount(bid_size_0, bid_price);
    ask_size_0 = instrument.getTradeAmount(ask_size_0, ask_price);
    
    // No hard position cap - let agent manage inventory risk through target_inventory
    // Only use config.max_leverage as an absolute safety limit (typically very high, e.g. 5x)
    
    // Check if placing bid is allowed: meet minimum size and absolute leverage limit
    bool can_place_bid = (bid_size_0 >= minAmount) && (leverage < config.max_leverage);
    
    // Check if placing ask is allowed: meet minimum size and absolute leverage limit
    bool can_place_ask = (ask_size_0 >= minAmount) && (leverage > -config.max_leverage);
    
    // Store last placed prices for diagnostics (0 indicates order not placed)
    last_mid_price = mid_price;
    last_bid_price = can_place_bid ? bid_price : 0.0;
    last_ask_price = can_place_ask ? ask_price : 0.0;
    
    // Place orders with hard position limits
    if (can_place_bid) {
        this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, bid_size_0);
    }
        
    if (can_place_ask) {
        this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, ask_size_0);
    }
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
