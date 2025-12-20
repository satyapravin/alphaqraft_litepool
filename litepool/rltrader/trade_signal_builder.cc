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

#include "trade_signal_builder.h"
#include <algorithm>
#include <cmath>

using namespace RLTrader;

TradeSignalBuilder::TradeSignalBuilder() 
    : buy_volume_ema_(0.0),
      sell_volume_ema_(0.0),
      prev_mid_price_(0.0),
      last_trade_timestamp_(0),
      max_volume_per_period_(1.0),
      max_total_volume_(100.0) {
}

void TradeSignalBuilder::reset() {
    recent_trades_.clear();
    buy_volume_ema_ = 0.0;
    sell_volume_ema_ = 0.0;
    prev_mid_price_ = 0.0;
    last_trade_timestamp_ = 0;  // Reset temporal tracking
    // Reset normalization parameters to initial values
    max_volume_per_period_ = 1.0;
    max_total_volume_ = 100.0;
}

TradeSignals TradeSignalBuilder::add_trades(
    const std::vector<Trade>& trades, 
    double mid_price,
    long long current_book_timestamp
) {
    TradeSignals signals;
    
    // Use provided book timestamp, or fallback to latest trade timestamp
    long long current_timestamp = current_book_timestamp;
    if (current_timestamp == 0 && !trades.empty()) {
        current_timestamp = trades.back().timestamp;
    }
    
    // CRITICAL: Clean up old trades FIRST using current timestamp (not per-trade)
    // This prevents unbounded growth and exponential slowdown
    while (!recent_trades_.empty() && current_timestamp > 0) {
        if ((current_timestamp - recent_trades_.front().timestamp) > TIME_WINDOW_US) {
            recent_trades_.pop_front();
        } else {
            break;  // Remaining trades are all within window
        }
    }
    
    // Add new trades to window and track temporal information
    for (const auto& trade : trades) {
        recent_trades_.push_back(trade);
        last_trade_timestamp_ = std::max(last_trade_timestamp_, trade.timestamp);
    }
    
    // Also enforce size limit as safety (shouldn't be needed if time window works)
    while (recent_trades_.size() > static_cast<size_t>(TRADE_WINDOW_SIZE * 2)) {
        recent_trades_.pop_front();  // Emergency cleanup if time window fails
    }
    
    // current_timestamp already set above
    
    // Compute statistics from window (100ms aggregated periods)
    // IMPORTANT: Each trade in recent_trades_ represents one 100ms aggregated period.
    // Each period will have either buy volume (uptick) or sell volume (downtick), never both zero.
    // If a period has no trades, it simply doesn't appear in recent_trades_.
    double buy_volume = 0.0;
    double sell_volume = 0.0;
    double total_volume = 0.0;
    int active_periods = 0;  // Number of 100ms periods with activity (each trade = 1 period)
    
    for (const auto& trade : recent_trades_) {
        // Each trade represents one 100ms period with either buy or sell activity
        if (trade.side == OrderSide::BUY) {
            buy_volume += trade.size;
        } else {
            sell_volume += trade.size;
        }
        total_volume += trade.size;
        active_periods++;  // Each trade = one 100ms period (uptick or downtick)
    }
    
    // Update adaptive normalization parameters
    if (active_periods > 0) {
        double avg_volume_per_period = total_volume / static_cast<double>(active_periods);
        if (avg_volume_per_period > max_volume_per_period_) {
            max_volume_per_period_ = avg_volume_per_period * 1.5;  // Add 50% margin
        }
    }
    
    if (total_volume > max_total_volume_) {
        max_total_volume_ = total_volume * 1.2;  // Add 20% margin
    }
    
    // 1. Volume imbalance: (buy - sell) / (buy + sell) - already in [-1, 1]
    // Since each 100ms period has either buy or sell, this reflects net directional pressure
    double total_vol = buy_volume + sell_volume;
    if (total_vol > 0) {
        // +1.0 = all buys (all upticks), -1.0 = all sells (all downticks), 0.0 = balanced
        signals.volume_imbalance = (buy_volume - sell_volume) / total_vol;
    } else {
        // No trades in window -> neutral (0.0), distinct from balanced (50/50) which would also be 0.0
        // But since we have no data, use 0.0 as neutral signal
        signals.volume_imbalance = 0.0;
    }
    // Clamp to ensure [-1, 1] (shouldn't be needed, but safety check)
    signals.volume_imbalance = std::clamp(signals.volume_imbalance, -1.0, 1.0);
    
    // 2. Trade intensity: volume per 100ms period (activity rate) - normalize to [0, 1] then map to [-1, 1]
    // For 100ms aggregates, this represents trading activity rate (volume per period)
    if (active_periods > 0) {
        double volume_per_period = total_volume / static_cast<double>(active_periods);
        double intensity_raw = volume_per_period / max_volume_per_period_;
        intensity_raw = std::clamp(intensity_raw, 0.0, 1.0);  // Clamp to [0, 1]
        signals.trade_intensity = 2.0 * intensity_raw - 1.0;  // Map [0, 1] -> [-1, 1]
    } else {
        signals.trade_intensity = -1.0;  // No activity
    }
    
    // 3. Price impact: price change per unit volume - already in [-1, 1] via tanh
    if (prev_mid_price_ > 0 && total_vol > 0) {
        double price_change = mid_price - prev_mid_price_;
        // Normalize by 0.1% of price, then use tanh to bound to [-1, 1]
        double impact_raw = price_change / (mid_price * 0.001);
        signals.price_impact = std::tanh(impact_raw);  // Bounded to [-1, 1]
    } else {
        signals.price_impact = 0.0;
    }
    prev_mid_price_ = mid_price;
    
    // 4. Buy/Sell pressure (EMA of volume momentum) - normalize as ratio of total EMA volume
    buy_volume_ema_ = EMA_ALPHA * buy_volume + (1.0 - EMA_ALPHA) * buy_volume_ema_;
    sell_volume_ema_ = EMA_ALPHA * sell_volume + (1.0 - EMA_ALPHA) * sell_volume_ema_;
    
    // Normalize as ratio: buy_ema / (buy_ema + sell_ema) for scale-invariance
    double total_ema_vol = buy_volume_ema_ + sell_volume_ema_;
    if (total_ema_vol > 0) {
        double buy_pressure_ratio = buy_volume_ema_ / total_ema_vol;  // [0, 1]
        double sell_pressure_ratio = sell_volume_ema_ / total_ema_vol;  // [0, 1]
        // Map [0, 1] -> [-1, 1] for consistency
        signals.buy_pressure = 2.0 * buy_pressure_ratio - 1.0;   // Map [0, 1] -> [-1, 1]
        signals.sell_pressure = 2.0 * sell_pressure_ratio - 1.0; // Map [0, 1] -> [-1, 1]
    } else {
        // No EMA volume -> -1.0 indicates absence of activity (distinct from balanced = 0.0)
        signals.buy_pressure = -1.0;  // No buy pressure
        signals.sell_pressure = -1.0; // No sell pressure
    }
    
    // 5. Buy/Sell volume - normalize as ratio of total volume: buy/(buy+sell) and sell/(buy+sell)
    // Since each 100ms period is either an uptick (buy) or downtick (sell), this reflects
    // the proportion of periods that were upticks vs downticks
    // This makes signals scale-invariant and directly interpretable (0.7 = 70% of periods were upticks)
    if (total_vol > 0) {
        double buy_ratio = buy_volume / total_vol;  // [0, 1] - proportion of volume that was buys
        double sell_ratio = sell_volume / total_vol;  // [0, 1] - proportion of volume that was sells
        // Map [0, 1] -> [-1, 1] for consistency with other signals
        // +1.0 = all buys (all upticks), -1.0 = all sells (all downticks), 0.0 = balanced
        signals.buy_volume = 2.0 * buy_ratio - 1.0;   // 0.0 -> -1.0, 0.5 -> 0.0, 1.0 -> 1.0
        signals.sell_volume = 2.0 * sell_ratio - 1.0; // 0.0 -> -1.0, 0.5 -> 0.0, 1.0 -> 1.0
    } else {
        // No trades in window -> -1.0 indicates absence of activity (distinct from balanced 50/50 = 0.0)
        signals.buy_volume = -1.0;  // No buy activity (no upticks)
        signals.sell_volume = -1.0; // No sell activity (no downticks)
    }
    
    // 6. Time since last trade - normalized time since last trade [-1, 1]
    // -1.0 = very recent (0s), 0.0 = moderate (5s), +1.0 = very old (10s+)
    // For 100ms aggregates, this tells us how long since the last period had activity
    if (last_trade_timestamp_ > 0 && current_timestamp > 0) {
        long long time_since = current_timestamp - last_trade_timestamp_;
        double time_ratio = std::min(1.0, static_cast<double>(time_since) / MAX_TIME_SINCE_TRADE_US);
        signals.time_since_last_trade = 2.0 * time_ratio - 1.0;  // Map [0, 1] -> [-1, 1]
    } else {
        signals.time_since_last_trade = -1.0;  // No trades yet -> very recent (or invalid)
    }
    
    return signals;
}

