#pragma once

#include <memory>
#include <cmath>

#include "base_exchange.h"
#include "orderbook.h"
#include "position.h"

namespace RLTrader {

    // Static configuration - limits and constraints (not learned)
    struct StrategyConfig {
        double max_leverage = 5.0;            // Maximum leverage limit (hard stop)
        double min_size_pct = 0.2;            // Minimum size as % of balance
        double max_size_pct = 2.2;            // Maximum size as % of balance
        double base_spread_bps = 5.0;         // Base spread in basis points (5 bps = 0.05%)
    };

    // RL action outputs - simplified 4-action space
    // Reduces redundancy and improves learning efficiency
    struct RLAction {
        // Spread control: [-1, 1] → multiplier [0.5, 1.5] on base spread
        double spread = 0.0;                  // Tighten (<0) or widen (>0) symmetric spread
        
        // Size control: [-1, 1] → [min_size_pct, max_size_pct] of balance
        double size = 0.0;                    // Small (<0) or large (>0) order size
        
        // Skew control: [-1, 1] → asymmetry based on inventory
        // Handles both spread skew AND size skew for inventory management
        // Positive skew = favor selling (widen bid, tighten ask, larger ask size)
        // Negative skew = favor buying (tighten bid, widen ask, larger bid size)
        double skew = 0.0;
        
        // Requote decision: >0 means requote, <=0 means keep existing orders
        double should_requote = 0.0;
    };

    class Strategy {
    public:
        Strategy(BaseInstrument& instr, BaseExchange& exch, const double& balance, 
                 int maxTicks, const StrategyConfig& config);
        
        void reset();

        // Main quoting interface - takes RL action and places orders
        void quote(const RLAction& action,
                   FixedVector<double, 20>& bid_prices, 
                   FixedVector<double, 20>& ask_prices);

        Position& getPosition() { return position; }
        StrategyConfig& getConfig() { return config; }
        
        // Process fills from exchange
        void next();

    protected:
        // Override this for different quoting strategies (e.g., GLFT)
        virtual std::pair<double, double> computeQuotePrices(
            const RLAction& action,
            double mid_price,
            double leverage,
            double tick_size);
        
        // Override this for different sizing strategies
        virtual std::pair<double, double> computeQuoteSizes(
            const RLAction& action,
            double init_balance);

    protected:
        BaseInstrument& instrument;
        BaseExchange& exchange;
        Position position;
        StrategyConfig config;
        int order_id;
        int max_ticks;
        
        // State variables
        double target_inventory_ema = 0.0;    // Smoothed target inventory
        
        // EMA smoothing for target inventory
        static constexpr double TARGET_EMA_ALPHA = 0.05;
    };
}
