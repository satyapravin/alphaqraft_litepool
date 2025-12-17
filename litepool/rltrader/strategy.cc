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
    // Scale action from [-1, 1] to [-0.1, 0.1] (representing -10% to +10% of balance)
    // This gives the RL agent full control over the target inventory range
    double scaled_target = target_inventory_action * 0.1;
    
    // EMA smoothing: target_inventory_ema = alpha * new_target + (1 - alpha) * old_target
    // TARGET_EMA_ALPHA = 0.05 means ~20 step half-life (smooth updates)
    target_inventory_ema = TARGET_EMA_ALPHA * scaled_target + 
                          (1.0 - TARGET_EMA_ALPHA) * target_inventory_ema;
    
    // Clamp the smoothed result to ensure it stays within bounds
    target_inventory_ema = std::clamp(target_inventory_ema, -0.1, 0.1);
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
    // Linear mapping: mult = (MAX+MIN)/2 + action * (MAX-MIN)/2
    // This gives symmetric, interpretable control
    double mid_mult = (MAX_SPREAD_MULT + MIN_SPREAD_MULT) / 2.0;
    double range_mult = (MAX_SPREAD_MULT - MIN_SPREAD_MULT) / 2.0;
    
    double bid_spread_mult = mid_mult + action.bid_spread * range_mult;
    double ask_spread_mult = mid_mult + action.ask_spread * range_mult;
    
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
    double inventory_error = leverage - target_inventory_ema;
    
    // Inventory-based skew: shift mid-point to reduce inventory
    // Negative sign: positive error (too long) → negative shift (quote lower)
    double inventory_skew = -inventory_error * INVENTORY_SKEW_MULT;
    
    // Hard position limit enforcement
    constexpr double MAX_POSITION_LEVERAGE = 0.1;  // ±10% of balance
    if (leverage > MAX_POSITION_LEVERAGE) {
        // Force stronger downward shift when over long limit
        inventory_skew = std::min(inventory_skew, -1.5);
    } else if (leverage < -MAX_POSITION_LEVERAGE) {
        // Force stronger upward shift when over short limit
        inventory_skew = std::max(inventory_skew, 1.5);
    }
    
    // Agent's additional skew control
    double action_skew = action.skew * ACTION_SKEW_MULT;
    
    // Total skew (in units of base_half_spread)
    // Reduced from ±3.0 to ±1.5 to prevent extreme asymmetric spreads
    // With max skew of 1.5, one side can be 1.5x wider while other is 1.5x tighter
    // This prevents the tight side from going below ~0.5 bps (still profitable)
    double total_skew = inventory_skew + action_skew;
    total_skew = std::clamp(total_skew, -1.5, 1.5);
    
    // Compute reservation price
    double reservation_price = mid_price + total_skew * base_half_spread;
    
    // === Step 6: Compute final quote prices ===
    double bid_price = reservation_price - bid_half_spread;
    double ask_price = reservation_price + ask_half_spread;
    
    // === Step 7: Safety checks ===
    // Minimum spread from mid-price: 1.0 bps (prevents adverse selection on too-tight quotes)
    // On $100k BTC, 1 bps = $10, which is a reasonable minimum edge
    double min_half_spread_from_mid = mid_price * 1.0 / 10000.0;  // 1.0 bps
    
    // Ensure bid is at least min_half_spread below mid
    if (mid_price - bid_price < min_half_spread_from_mid) {
        bid_price = mid_price - min_half_spread_from_mid;
    }
    
    // Ensure ask is at least min_half_spread above mid
    if (ask_price - mid_price < min_half_spread_from_mid) {
        ask_price = mid_price + min_half_spread_from_mid;
    }
    
    // Ensure minimum spread of 1 tick between bid and ask
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
    assert(action.skew       >= -1.0001 && action.skew       <= 1.0001);
    
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
    // Prevents the bug where bid ends up above mid in wide-spread markets
    double min_spread_from_mid = mid_price * 1.0 / 10000.0;  // 1 bps minimum
    if (bid_price > mid_price - min_spread_from_mid) {
        bid_price = mid_price - min_spread_from_mid;
        bid_price = std::floor(bid_price / tick_size) * tick_size;
    }
    if (ask_price < mid_price + min_spread_from_mid) {
        ask_price = mid_price + min_spread_from_mid;
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
    
    // Hard position limit: ±10% of balance (leverage ±0.1)
    constexpr double MAX_POSITION_LEVERAGE = 0.1;
    
    // Calculate actual position value using net amount (not leverage * initBalance which is incorrect)
    // leverage = position_value / equity, but we need position_value directly
    double net_amount = position.getNetAmount();
    double current_position_value = net_amount * mid_price;
    double bid_order_value = bid_size_0 * bid_price;
    double ask_order_value = ask_size_0 * ask_price;
    double estimated_new_leverage_bid = (current_position_value + bid_order_value) / initBalance;
    double estimated_new_leverage_ask = (current_position_value - ask_order_value) / initBalance;
    
    // Check if placing bid is allowed:
    // - Must meet minimum size and leverage limits
    // - If already over +10% limit, disallow bids (they increase long position)
    // - Otherwise, only allow if bid won't exceed +10% limit
    bool can_place_bid = (bid_size_0 >= minAmount) && (leverage < config.max_leverage);
    if (can_place_bid) {
        if (leverage > MAX_POSITION_LEVERAGE) {
            // Already over +10% limit - disallow bids (they increase long position)
            can_place_bid = false;
        } else {
            // Within limit - check if bid would exceed it
            can_place_bid = (estimated_new_leverage_bid <= MAX_POSITION_LEVERAGE);
        }
    }
    
    // Check if placing ask is allowed:
    // - Must meet minimum size and leverage limits
    // - If already over -10% limit, disallow asks (they increase short position)
    // - Otherwise, only allow if ask won't exceed -10% limit
    bool can_place_ask = (ask_size_0 >= minAmount) && (leverage > -config.max_leverage);
    if (can_place_ask) {
        if (leverage < -MAX_POSITION_LEVERAGE) {
            // Already over -10% limit - disallow asks (they increase short position)
            can_place_ask = false;
        } else {
            // Within limit - check if ask would exceed it
            can_place_ask = (estimated_new_leverage_ask >= -MAX_POSITION_LEVERAGE);
        }
    }
    
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
