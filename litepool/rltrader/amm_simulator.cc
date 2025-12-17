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
    // For V3: total_value = L * [2*sqrt(p) - sqrt(pa) - p/sqrt(pb)]
    double sqrt_p = std::sqrt(initial_price);
    double denominator = 2.0 * sqrt_p - sqrt_pa_ - initial_price / sqrt_pb_;
    if (denominator > 0) {
        liquidity_ = LIQUIDITY_USD / denominator;
    } else {
        liquidity_ = LIQUIDITY_USD;  // Fallback
    }
    
    // Compute initial reserves
    computeReserves(initial_price);
    
    // Reset flow tracking
    recent_flows_.clear();
    window_buy_vol_ = 0.0;
    window_sell_vol_ = 0.0;
    
    // Reset EMA-based net flow tracking
    net_flow_ema_ = 0.0;
    net_flow_magnitude_ema_ = initial_price * 0.001;  // Start with small magnitude estimate
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
    double sqrt_center = std::sqrt(center_price_ema_);
    double denominator = 2.0 * sqrt_center - sqrt_pa_ - center_price_ema_ / sqrt_pb_;
    if (denominator > 0) {
        liquidity_ = LIQUIDITY_USD / denominator;
    }
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
    
    // Store old reserves (computed with CURRENT liquidity, not old liquidity)
    double old_y = reserve_y_;
    
    // Compute new reserves at target price
    // IMPORTANT: Use the same liquidity that was used for old reserves
    // This ensures trade size reflects actual price movement, not liquidity changes
    computeReserves(target_price);
    current_price_ = target_price;
    
    // Trade size is change in quote reserves
    // Positive dy = someone bought base asset (bullish flow)
    // Negative dy = someone sold base asset (bearish flow)
    double dy = reserve_y_ - old_y;
    
    return dy;
}

void AmmV3Simulator::updateFlowTracking(double trade_size) {
    // Create flow entry for this step
    StepFlow step_flow;
    if (trade_size > 0) {
        step_flow.buy_vol = trade_size;
    } else if (trade_size < 0) {
        step_flow.sell_vol = -trade_size;
    }
    
    // Add to window
    recent_flows_.push_back(step_flow);
    window_buy_vol_ += step_flow.buy_vol;
    window_sell_vol_ += step_flow.sell_vol;
    
    // Maintain window size (FLOW_WINDOW steps, not trades)
    while (recent_flows_.size() > static_cast<size_t>(FLOW_WINDOW)) {
        const StepFlow& old_flow = recent_flows_.front();
        window_buy_vol_ -= old_flow.buy_vol;
        window_sell_vol_ -= old_flow.sell_vol;
        recent_flows_.pop_front();
    }
    
    // Update EMA-based net flow signal
    // This approach:
    // 1. Computes a normalized trade direction for this step
    // 2. Uses EMA to smooth it over time
    // 3. Avoids the issues with cumulative normalization
    
    double trade_magnitude = std::abs(trade_size);
    
    // Update magnitude EMA (for normalization)
    net_flow_magnitude_ema_ = NET_FLOW_EMA_ALPHA * trade_magnitude + 
                               (1.0 - NET_FLOW_EMA_ALPHA) * net_flow_magnitude_ema_;
    
    // Ensure minimum magnitude to prevent division issues
    double magnitude_for_norm = std::max(net_flow_magnitude_ema_, current_price_ * 0.0001);
    
    // Normalize this step's trade: [-1, 1] range
    double normalized_trade = 0.0;
    if (trade_magnitude > 0) {
        normalized_trade = trade_size / magnitude_for_norm;
        normalized_trade = std::clamp(normalized_trade, -1.0, 1.0);
    }
    
    // EMA update for net flow signal
    // This creates a momentum indicator that responds to recent flow direction
    net_flow_ema_ = NET_FLOW_EMA_ALPHA * normalized_trade + 
                    (1.0 - NET_FLOW_EMA_ALPHA) * net_flow_ema_;
    
    // Decay toward zero when no trades (mean-reverting)
    // This prevents the signal from getting stuck at extremes
    net_flow_ema_ *= 0.995;  // Slow decay
}

AmmFlowSignals AmmV3Simulator::step(double market_price) {
    AmmFlowSignals signals{0.0, 0.0, 0.0};
    
    if (!initialized_) {
        reset(market_price);
        return signals;
    }
    
    // === IMPORTANT: Simulate arbitrage BEFORE updating range ===
    // This ensures trade size reflects actual price movement with consistent liquidity
    double trade_size = simulateArbitrage(market_price);
    
    // === Now update the rolling range for next step ===
    // The new liquidity will be used for the next step's reserve calculations
    updateRollingRange(market_price);
    
    // Recompute reserves with updated range (for accurate reserve getters)
    computeReserves(market_price);
    
    // Update flow tracking
    updateFlowTracking(trade_size);
    
    // === Compute normalized signals ===
    
    // 1. Net flow: EMA-based momentum signal
    //    Positive = recent net buying pressure, Negative = recent net selling pressure
    //    Uses EMA smoothing which naturally prevents signal decay issues
    signals.net_flow = std::clamp(net_flow_ema_, -1.0, 1.0);
    
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
