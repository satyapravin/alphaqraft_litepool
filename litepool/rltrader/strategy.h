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
#include <deque>
#include <vector>

#include "base_exchange.h"
#include "orderbook.h"
#include "position.h"

namespace RLTrader {

    // Static configuration - limits and constraints (not learned)
    struct StrategyConfig {
        double max_leverage = 2;            // Maximum leverage limit (hard stop)
        double min_size_pct = 0.2;            // Minimum size as % of balance
        double max_size_pct = 2.2;            // Maximum size as % of balance
        double base_spread_bps = 5.0;         // Base spread in basis points (5 bps = 0.05%)
    };

    // RL action outputs - 4-action space (bid_spread, ask_spread, target_inventory, risk_aversion)
    struct RLAction {
        // Bid spread control: [0, 1] → multiplier on base spread for bid side
        double bid_spread = 0.0;
        
        // Ask spread control: [0, 1] → multiplier on base spread for ask side
        double ask_spread = 0.0;
        
        // Target inventory: [-1, 1] → scaled to ±target_range target leverage
        // Positive = target long position, Negative = target short position
        double target_inventory = 0.0;
        
        // Risk aversion parameter (γ) for Avellaneda-Stoikov model: [0, 1]
        // Higher γ = more risk averse = wider spreads and stronger inventory adjustment
        double risk_aversion = 0.5;
        
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
        double getVolatility() const { return vol_2min_; }
        double getTargetInventory() const { return target_inventory_ema; }
        double getRiskAversion() const { return risk_aversion_ema; }
        bool hitLeverageLimit() const { return hit_leverage_limit_; }
        int getStepsSinceLastFill() const { return steps_since_last_fill_; }
        
        // Update smoothed target inventory and risk aversion (EMA smoothing to prevent flickering)
        void updateTargetInventory(double target_inventory_action, double risk_aversion_action);
        
        // Process fills from exchange
        void next();
        
        // Check if proposed quotes differ significantly from current quotes
        // Returns true if we should requote (prices differ by more than tick_threshold ticks)
        bool shouldRequote(const RLAction& action,
                          FixedVector<double, 20>& bid_prices,
                          FixedVector<double, 20>& ask_prices,
                          double tick_threshold = 2.0);

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
        double target_inventory_ema = 0.0;    // Smoothed target inventory (clamped to [-target_range, +target_range])
        double risk_aversion_ema = 0.5;       // Smoothed risk aversion parameter (γ) for A-S model [0, 1]
        
        // Track last placed quote prices for diagnostics
        double last_bid_price = 0.0;
        double last_ask_price = 0.0;
        double last_mid_price = 0.0;
        
        // Volatility tracking (2-minute window)
        std::deque<double> price_history_;    // Price history for 2-minute volatility calculation
        double vol_2min_ = 0.0;               // 2-minute volatility estimate
        static constexpr int VOL_WINDOW_STEPS = 240;  // 2 minutes = 240 steps at 0.5s per step
        
        // Inventory skew urgency tracking (time-based)
        double prev_inventory_error_ = 0.0;   // Previous inventory error for tracking persistence
        int steps_away_from_target_ = 0;      // Steps since we've been away from target (urgency)
        static constexpr double URGENCY_TIME_THRESHOLD = 0.01;  // Error threshold to consider "away from target"
        
        // Leverage limit tracking
        bool hit_leverage_limit_ = false;     // True if leverage hit ±1.0 and quoting stopped
        
        // Fill tracking for observations
        int steps_since_last_fill_ = 0;       // Steps since last fill (for time_since_last_fill signal)
        long prev_trade_count_ = 0;           // Trade count at last step (to detect new fills)
        
        // Tracking for netAmount change detection (per-instance, not static)
        double last_netAmount_ = 0.0;         // Last netAmount value (for detecting changes without fills)
        long last_trades_ = 0;                // Last trade count (for detecting changes without fills)
        bool first_call_ = true;               // First call flag (for detecting changes without fills)
        
        // Model parameters
        static constexpr double TARGET_EMA_ALPHA = 0.0058;   // Target inventory smoothing (120 step half-life = 60 sec)
        // Alpha = 0.0058 gives half-life of ~120 steps: 0.5 = (1 - alpha)^n → n ≈ 120 
        static constexpr double VOL_SPREAD_MULT = 50.0;      // How much volatility widens spread
        static constexpr double INVENTORY_SKEW_MULT = 0.5;  // Conservative skew: max 50% of spread can be adjusted (preserves spread capture)
        static constexpr double MAX_SPREAD_MULT = 50.0;      // Maximum spread multiplier from action (allows huge spreads for liquidity crunches)
        static constexpr double MIN_SPREAD_MULT = 0.5;       // Minimum spread multiplier (0.5x base = 1.5bps floor)
    };
}
