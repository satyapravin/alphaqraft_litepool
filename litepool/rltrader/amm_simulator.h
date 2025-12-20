#pragma once
#include <cmath>
#include <deque>
#include <algorithm>

namespace RLTrader {

/**
 * AMM V3-style simulator with ROLLING concentrated liquidity range.
 * Simulates a Uniswap V3-like LP position that follows the market price.
 * 
 * The LP range smoothly tracks the market price using EMA, ensuring
 * the position is always active and generating meaningful flow signals.
 * 
 * Used to generate inventory flow signals from market data:
 * - net_flow: Cumulative (buys - sells) in quote currency
 * - flow_imbalance: buy_vol / total_vol over recent window
 * - inventory_delta: Normalized change in LP holdings relative to centered position
 */
struct AmmFlowSignals {
    double net_flow;           // EMA-based net flow signal (momentum indicator)
    double flow_imbalance;     // buy_vol / total_vol over window [-1, 1] centered
    double inventory_delta;    // Position in range: -1 (all base) to +1 (all quote)
    double cumulative_flow;    // Raw cumulative (buys - sells) in USD - to be normalized by balance
};

class AmmV3Simulator {
public:
    // Configuration
    static constexpr double RANGE_WIDTH = 0.40;      // 40% total range (±20%)
    static constexpr double FEE_RATE = 0.0005;       // 5 bps (0.05% pool)
    static constexpr int FLOW_WINDOW = 100;          // Steps for imbalance calc
    static constexpr double LIQUIDITY_USD = 100000;  // Virtual liquidity
    static constexpr double RANGE_EMA_ALPHA = 0.02;  // Rolling range smoothing (slower = smoother)
    static constexpr double NET_FLOW_EMA_ALPHA = 0.05;  // EMA for net flow signal
    
    AmmV3Simulator() : initialized_(false) {}
    
    /**
     * Initialize/reset the simulator with a new price.
     * Sets up concentrated liquidity range around the price.
     */
    void reset(double initial_price);
    
    /**
     * Clear initialized state so next step() will auto-initialize.
     * Call this at episode boundaries.
     * Also clears all containers and resets state to prevent accumulation.
     */
    void clear() {
        initialized_ = false;
        recent_flows_.clear();
        window_buy_vol_ = 0.0;
        window_sell_vol_ = 0.0;
        net_flow_ema_ = 0.0;
        net_flow_magnitude_ema_ = 0.0;
        cumulative_flow_ = 0.0;
    }
    
    /**
     * Step the simulator with new market price.
     * Simulates arbitrage and tracks flow.
     * 
     * @param market_price Current mid price from orderbook
     * @return Flow signals for RL observation
     */
    AmmFlowSignals step(double market_price);
    
    // Getters
    double getReserveX() const { return reserve_x_; }
    double getReserveY() const { return reserve_y_; }
    double getCurrentPrice() const { return current_price_; }
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_;
    
    // Rolling range center (EMA of market price)
    double center_price_ema_;
    
    // Position parameters (V3 concentrated liquidity)
    double liquidity_;     // L in V3 math (constant value)
    double price_lower_;   // pa (lower bound of range) - rolling
    double price_upper_;   // pb (upper bound of range) - rolling
    double sqrt_pa_;       // cached sqrt(pa)
    double sqrt_pb_;       // cached sqrt(pb)
    
    // Current state
    double reserve_x_;     // Base asset (e.g., BTC)
    double reserve_y_;     // Quote asset (e.g., USD)
    double current_price_;
    
    // Flow tracking (per-step, not per-trade)
    struct StepFlow {
        double buy_vol = 0.0;
        double sell_vol = 0.0;
    };
    std::deque<StepFlow> recent_flows_;  // Track buy/sell per step
    double window_buy_vol_;               // Sum of buys in window
    double window_sell_vol_;              // Sum of sells in window
    
    // EMA-based net flow tracking (avoids range normalization issues)
    double net_flow_ema_;                 // EMA of trade direction signal
    double net_flow_magnitude_ema_;       // EMA of trade magnitude for normalization
    double cumulative_flow_;              // Raw cumulative (buys - sells) in USD
    
    /**
     * Update rolling range center using EMA of market price.
     * Recomputes range bounds [center * 0.8, center * 1.2].
     */
    void updateRollingRange(double market_price);
    
    /**
     * Compute reserves given current price using V3 math.
     * For price p in range [pa, pb] with liquidity L:
     *   x = L * (sqrt(pb) - sqrt(p)) / (sqrt(p) * sqrt(pb))
     *   y = L * (sqrt(p) - sqrt(pa))
     */
    void computeReserves(double price);
    
    /**
     * Simulate arbitrage trade to move AMM price toward target.
     * Returns the trade size (positive = buy, negative = sell).
     */
    double simulateArbitrage(double target_price);
    
    /**
     * Update flow tracking with a new trade.
     */
    void updateFlowTracking(double trade_size);
    
    /**
     * Compute position within range: -1 (all base) to +1 (all quote).
     */
    double computeRangePosition(double price);
};

} // namespace RLTrader

