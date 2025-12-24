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

    // RL action outputs - 4-action space (skew removed - determined automatically from inventory error)
    struct RLAction {
        // Bid spread control: [-1, 1] → multiplier on base spread for bid side
        double bid_spread = 0.0;             // Tighten (<0) or widen (>0) bid spread
        
        // Ask spread control: [-1, 1] → multiplier on base spread for ask side
        double ask_spread = 0.0;             // Tighten (<0) or widen (>0) ask spread
        
        // Target inventory: [-1, 1] → scaled to ±10% target leverage
        // Positive = target long position, Negative = target short position
        // Skew is automatically computed from (current_leverage - target) to push toward target
        double target_inventory = 0.0;
        
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
        
        // Get last placed quote prices for diagnostics
        double getLastBidPrice() const { return last_bid_price; }
        double getLastAskPrice() const { return last_ask_price; }
        double getLastMidPrice() const { return last_mid_price; }
        double getVolatility() const { return realized_vol; }
        double getTargetInventory() const { return target_inventory_ema; }
        bool hitLeverageLimit() const { return hit_leverage_limit_; }
        int getStepsSinceLastFill() const { return steps_since_last_fill_; }
        
        // Update smoothed target inventory (EMA smoothing to prevent flickering)
        void updateTargetInventory(double target_inventory_action);
        
        // Process fills from exchange
        void next();

    protected:
        // Avellaneda-Stoikov inspired quoting model
        virtual std::pair<double, double> computeQuotePrices(
            const RLAction& action,
            double mid_price,
            double leverage,
            double tick_size);
        
        // Compute order sizes
        virtual std::pair<double, double> computeQuoteSizes(
            const RLAction& action,
            double init_balance);
        
        // Update volatility estimate from mid-price changes
        void updateVolatility(double mid_price);

    protected:
        BaseInstrument& instrument;
        BaseExchange& exchange;
        Position position;
        StrategyConfig config;
        int order_id;
        int max_ticks;
        
        // State variables
        double target_inventory_ema = 0.0;    // Smoothed target inventory (clamped to [-0.1, +0.1])
        
        // Track last placed quote prices for diagnostics
        double last_bid_price = 0.0;
        double last_ask_price = 0.0;
        double last_mid_price = 0.0;
        
        // Volatility tracking (EMA of squared returns)
        double prev_mid_price = 0.0;
        double realized_vol = 0.0;            // EMA volatility estimate
        
        // Leverage limit tracking
        bool hit_leverage_limit_ = false;     // True if leverage hit ±1.0 and quoting stopped
        
        // Fill tracking for observations
        int steps_since_last_fill_ = 0;       // Steps since last fill (for time_since_last_fill signal)
        long prev_trade_count_ = 0;           // Trade count at last step (to detect new fills)
        
        // Model parameters
        static constexpr double TARGET_EMA_ALPHA = 0.001;     // Target inventory smoothing (~600 step half-life = 5 min)
        static constexpr double VOL_EMA_ALPHA = 0.01;        // Volatility EMA (~100 sample half-life)
        static constexpr double VOL_SPREAD_MULT = 50.0;      // How much volatility widens spread
        static constexpr double INVENTORY_SKEW_MULT = 50.0;  // Reduced: smoother quote shifts, agent spreads respected more
        // REMOVED: ACTION_SKEW_MULT - skew is now fully determined by inventory error
        // Agent controls inventory via target_inventory action only (no conflicting skew action)
        static constexpr double MAX_SPREAD_MULT = 50.0;      // Maximum spread multiplier from action (allows huge spreads for liquidity crunches)
        static constexpr double MIN_SPREAD_MULT = 0.5;       // Minimum spread multiplier (0.5x base = 1.5bps floor)
    };
}
