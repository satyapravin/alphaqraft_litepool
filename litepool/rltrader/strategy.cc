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
    this->exchange.fetchPosition(initQty, avgPrice, false);
    this->position.reset(initQty, avgPrice);
    this->order_id = 0;
}

std::pair<double, double> Strategy::computeQuotePrices(
    const RLAction& action,
    double mid_price,
    double leverage,
    double tick_size) {
    
    // Simplified 4-action approach:
    // - action.spread: symmetric spread control [-1, 1] → [0.5, 1.5] multiplier
    // - action.skew: asymmetry control [-1, 1] → bid/ask adjustment
    
    // Base half-spread from config (e.g., 2 bps = 0.02% = 0.0002)
    double half_spread = mid_price * config.base_spread_bps / 10000.0 / 2.0;
    
    // Symmetric spread multiplier: [-1, 1] → [0.5, 1.5]
    double spread_mult = 1.0 + action.spread * 0.5;
    
    // Skew creates asymmetry:
    // - Positive skew: widen bid (less aggressive buy), tighten ask (more aggressive sell)
    // - Negative skew: tighten bid (more aggressive buy), widen ask (less aggressive sell)
    // Also incorporates current inventory to help with mean reversion
    double skew_from_action = action.skew * 0.3;  // Action contributes 30% skew
    double skew_from_inventory = leverage * 0.2;  // Inventory contributes 20% auto-skew
    double total_skew = skew_from_action + skew_from_inventory;
    total_skew = std::clamp(total_skew, -0.5, 0.5);  // Limit total skew
    
    // Apply spread and skew
    double bid_mult = spread_mult * (1.0 + total_skew);   // Widen bid if positive skew
    double ask_mult = spread_mult * (1.0 - total_skew);   // Tighten ask if positive skew
    
    double bid_price = mid_price - half_spread * bid_mult;
    double ask_price = mid_price + half_spread * ask_mult;
    
    // Ensure minimum spread of 1 tick
    if (ask_price - bid_price < tick_size) {
        double center = (bid_price + ask_price) / 2.0;
        bid_price = center - tick_size / 2.0;
        ask_price = center + tick_size / 2.0;
    }
    
    // Round to tick size
    bid_price = std::floor(bid_price / tick_size) * tick_size;
    ask_price = std::ceil(ask_price / tick_size) * tick_size;
    
    return {bid_price, ask_price};
}

std::pair<double, double> Strategy::computeQuoteSizes(
    const RLAction& action,
    double init_balance) {
    
    // Simplified: symmetric base size with skew-based asymmetry
    // action.size: [-1, 1] → [min_size_pct, max_size_pct]
    // action.skew: creates size asymmetry (positive = larger ask, smaller bid)
    
    double size_range = config.max_size_pct - config.min_size_pct;
    double base_size_pct = config.min_size_pct + (1.0 + action.size) * 0.5 * size_range;
    
    // Skew adjusts sizes: positive skew = want to sell more, buy less
    double skew_adj = action.skew * 0.3;  // ±30% size adjustment from skew
    double bid_size_pct = base_size_pct * (1.0 - skew_adj);
    double ask_size_pct = base_size_pct * (1.0 + skew_adj);
    
    // Clamp to valid range
    bid_size_pct = std::clamp(bid_size_pct, config.min_size_pct, config.max_size_pct);
    ask_size_pct = std::clamp(ask_size_pct, config.min_size_pct, config.max_size_pct);
    
    return {bid_size_pct / 100.0 * init_balance, 
            ask_size_pct / 100.0 * init_balance};
}

void Strategy::quote(const RLAction& action,
                     FixedVector<double, 20>& bid_prices,
                     FixedVector<double, 20>& ask_prices) {
    // Validate RL outputs are in expected range [-1, 1] (simplified 4-action space)
    assert(action.spread >= -1.0001 && action.spread <= 1.0001);
    assert(action.size   >= -1.0001 && action.size   <= 1.0001);
    assert(action.skew   >= -1.0001 && action.skew   <= 1.0001);
    
    // Early exit if prices invalid
    if (bid_prices[0] < 0.0001 || ask_prices[0] < 0.0001) return;
    
    auto tick_size = instrument.getTickSize();
    auto minAmount = instrument.getMinAmount();
    auto posInfo = position.getPositionInfo(bid_prices[0], ask_prices[0]);
    auto leverage = posInfo.leverage;
    auto initBalance = position.getInitialBalance();
    auto mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
    
    // Compute quote prices and sizes using (potentially overridden) methods
    auto [bid_price, ask_price] = computeQuotePrices(action, mid_price, leverage, tick_size);
    auto [bid_size_0, ask_size_0] = computeQuoteSizes(action, initBalance);
    
    // Ensure quotes don't cross the book, but allow quoting inside the spread for better fills
    double best_bid = bid_prices[0];
    double best_ask = ask_prices[0];
    
    // Don't cross: bid must be < best_ask, ask must be > best_bid
    // But allow inside spread: bid can be > best_bid, ask can be < best_ask
    if (bid_price >= best_ask) bid_price = best_ask - tick_size;  // Don't cross
    if (ask_price <= best_bid) ask_price = best_bid + tick_size;  // Don't cross
    
    // Ensure valid spread
    if (bid_price >= ask_price) {
        // Fallback: quote at edges if computed prices are invalid
        bid_price = best_bid;
        ask_price = best_ask;
    }
    
    // Cancel existing orders before placing new ones
    exchange.cancelOrders();
    
    // Convert to trade amounts (rounded to minAmount)
    bid_size_0 = instrument.getTradeAmount(bid_size_0, bid_price);
    ask_size_0 = instrument.getTradeAmount(ask_size_0, ask_price);
    
    // Place orders with leverage limits
    if (bid_size_0 >= minAmount && leverage < config.max_leverage) {
        this->exchange.quote(std::to_string(++order_id), OrderSide::BUY, bid_price, bid_size_0);
    }
        
    if (ask_size_0 >= minAmount && leverage > -config.max_leverage) {
        this->exchange.quote(std::to_string(++order_id), OrderSide::SELL, ask_price, ask_size_0);
    }
}

void Strategy::next() {
    auto fills = exchange.getFills();
    for(const auto& order: fills) {
        position.onFill(order);
    }
}
