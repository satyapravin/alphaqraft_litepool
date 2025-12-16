#include "amm_simulator.h"
#include <cmath>
#include <algorithm>

namespace RLTrader {

void AmmV3Simulator::reset(double initial_price) {
    if (initial_price <= 0) return;
    
    initialized_ = true;
    current_price_ = initial_price;
    center_price_ema_ = initial_price;
    
    // Set up 40% range: [price * 0.8, price * 1.2]
    price_lower_ = initial_price * (1.0 - RANGE_WIDTH / 2.0);
    price_upper_ = initial_price * (1.0 + RANGE_WIDTH / 2.0);
    sqrt_pa_ = std::sqrt(price_lower_);
    sqrt_pb_ = std::sqrt(price_upper_);
    
    // Calculate liquidity L from desired USD value
    // This stays constant - we're simulating a fixed-value LP position
    double sqrt_p = std::sqrt(initial_price);
    liquidity_ = LIQUIDITY_USD / (2.0 * (sqrt_p - sqrt_pa_) + 
                                   2.0 * initial_price * (sqrt_pb_ - sqrt_p) / (sqrt_p * sqrt_pb_));
    
    // Compute initial reserves
    computeReserves(initial_price);
    
    // Reset flow tracking
    cumulative_net_flow_ = 0.0;
    recent_buys_.clear();
    recent_sells_.clear();
    window_buy_vol_ = 0.0;
    window_sell_vol_ = 0.0;
    
    // Reset normalization (will adapt over time)
    max_observed_flow_ = initial_price * 0.01;  // Start with 1% of price
}

void AmmV3Simulator::updateRollingRange(double market_price) {
    // Update center price EMA (smooth tracking of market)
    center_price_ema_ = RANGE_EMA_ALPHA * market_price + 
                        (1.0 - RANGE_EMA_ALPHA) * center_price_ema_;
    
    // Recompute range bounds around the smoothed center
    price_lower_ = center_price_ema_ * (1.0 - RANGE_WIDTH / 2.0);
    price_upper_ = center_price_ema_ * (1.0 + RANGE_WIDTH / 2.0);
    sqrt_pa_ = std::sqrt(price_lower_);
    sqrt_pb_ = std::sqrt(price_upper_);
    
    // Recalculate liquidity to maintain constant USD value
    // This ensures consistent signal magnitude as price moves
    double sqrt_center = std::sqrt(center_price_ema_);
    liquidity_ = LIQUIDITY_USD / (2.0 * (sqrt_center - sqrt_pa_) + 
                                   2.0 * center_price_ema_ * (sqrt_pb_ - sqrt_center) / (sqrt_center * sqrt_pb_));
}

void AmmV3Simulator::computeReserves(double price) {
    if (price <= 0) return;
    
    double sqrt_p = std::sqrt(price);
    
    if (price <= price_lower_) {
        // All in base asset (x), no quote asset
        reserve_x_ = liquidity_ * (sqrt_pb_ - sqrt_pa_) / (sqrt_pa_ * sqrt_pb_);
        reserve_y_ = 0.0;
    } else if (price >= price_upper_) {
        // All in quote asset (y), no base asset
        reserve_x_ = 0.0;
        reserve_y_ = liquidity_ * (sqrt_pb_ - sqrt_pa_);
    } else {
        // Price in range - have both assets
        // V3 formulas for concentrated liquidity:
        // x = L * (sqrt(pb) - sqrt(p)) / (sqrt(p) * sqrt(pb))
        // y = L * (sqrt(p) - sqrt(pa))
        reserve_x_ = liquidity_ * (sqrt_pb_ - sqrt_p) / (sqrt_p * sqrt_pb_);
        reserve_y_ = liquidity_ * (sqrt_p - sqrt_pa_);
    }
}

double AmmV3Simulator::computeRangePosition(double price) {
    // Returns position within range: -1 (at lower, all base) to +1 (at upper, all quote)
    // 0 means price is at center of range
    if (price <= price_lower_) return -1.0;
    if (price >= price_upper_) return 1.0;
    
    // Linear interpolation within range, centered at 0
    double range_fraction = (price - price_lower_) / (price_upper_ - price_lower_);
    return 2.0 * range_fraction - 1.0;  // Map [0,1] to [-1,1]
}

double AmmV3Simulator::simulateArbitrage(double target_price) {
    if (!initialized_ || target_price <= 0) return 0.0;
    
    // If price hasn't changed significantly, no arbitrage
    double price_diff = target_price - current_price_;
    if (std::abs(price_diff) < current_price_ * 0.0001) {
        return 0.0;
    }
    
    // Store old reserves
    double old_y = reserve_y_;
    
    // Compute new reserves at target price (within current range)
    computeReserves(target_price);
    current_price_ = target_price;
    
    // Trade size is change in quote reserves
    // Positive dy = someone bought base asset (bullish flow)
    // Negative dy = someone sold base asset (bearish flow)
    double dy = reserve_y_ - old_y;
    
    return dy;
}

void AmmV3Simulator::updateFlowTracking(double trade_size) {
    // Update cumulative flow
    cumulative_net_flow_ += trade_size;
    
    // Update windowed tracking
    if (trade_size > 0) {
        recent_buys_.push_back(trade_size);
        window_buy_vol_ += trade_size;
    } else if (trade_size < 0) {
        recent_sells_.push_back(-trade_size);
        window_sell_vol_ += (-trade_size);
    }
    
    // Maintain window size
    while (recent_buys_.size() > FLOW_WINDOW) {
        window_buy_vol_ -= recent_buys_.front();
        recent_buys_.pop_front();
    }
    while (recent_sells_.size() > FLOW_WINDOW) {
        window_sell_vol_ -= recent_sells_.front();
        recent_sells_.pop_front();
    }
    
    // Update max observed for normalization
    max_observed_flow_ = std::max(max_observed_flow_, std::abs(cumulative_net_flow_));
}

AmmFlowSignals AmmV3Simulator::step(double market_price) {
    AmmFlowSignals signals{0.0, 0.0, 0.0};
    
    if (!initialized_) {
        reset(market_price);
        return signals;
    }
    
    // === Rolling range: smoothly follow market price ===
    updateRollingRange(market_price);
    
    // Simulate arbitrage trade (within rolling range)
    double trade_size = simulateArbitrage(market_price);
    
    // Update flow tracking
    updateFlowTracking(trade_size);
    
    // === Compute normalized signals ===
    
    // 1. Net flow: cumulative flow normalized to [-1, 1]
    //    Positive = net buying pressure, Negative = net selling pressure
    if (max_observed_flow_ > 0) {
        signals.net_flow = std::clamp(cumulative_net_flow_ / max_observed_flow_, -1.0, 1.0);
    }
    
    // 2. Flow imbalance: (buy - sell) / (buy + sell) over recent window
    //    +1 = all buys, -1 = all sells, 0 = balanced
    double total_vol = window_buy_vol_ + window_sell_vol_;
    if (total_vol > 0) {
        signals.flow_imbalance = (window_buy_vol_ - window_sell_vol_) / total_vol;
    }
    
    // 3. Inventory delta: position within rolling range
    //    -1 = price at lower bound (LP holds all base asset)
    //    +1 = price at upper bound (LP holds all quote asset)
    //     0 = price at center (balanced 50/50)
    signals.inventory_delta = computeRangePosition(market_price);
    
    return signals;
}

} // namespace RLTrader
