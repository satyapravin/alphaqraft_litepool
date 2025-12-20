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

#include "market_signal_builder.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include "rl_macros.h"

using namespace RLTrader;

MarketSignalBuilder::MarketSignalBuilder()
    : snapshot_buffer_{},
      snapshot_count_(0),
      snapshot_index_(0),
      ofi_ema_(0),
      vol_baseline_ema_(0),
      mid_price_history_(),
      spread_signals_(std::make_unique<spread_signal_repository>()),
      volume_signals_(std::make_unique<volume_signal_repository>()),
      volatility_signals_(std::make_unique<volatility_signal_repository>())
{
}

double MarketSignalBuilder::sigmoid_normalize(double x, double midpoint, double steepness) {
    // Maps x to [0, 1] using sigmoid: 1 / (1 + exp(-steepness * (x - midpoint)))
    // midpoint: value where output = 0.5
    // steepness: controls transition sharpness
    double z = steepness * (x - midpoint);
    // Clamp to prevent overflow
    z = std::clamp(z, -20.0, 20.0);
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<double> MarketSignalBuilder::add_book(OrderBook& book) {
    // Compute snapshot from current orderbook
    BookSnapshot current = compute_snapshot(book);
    
    // Add mid price to history for volatility calculation
    if (current.mid_price > 0) {
        mid_price_history_.push_back(current.mid_price);
        // Keep only VOL_LONG_WINDOW + 1 prices (need +1 for returns)
        while (mid_price_history_.size() > static_cast<size_t>(VOL_LONG_WINDOW + 1)) {
            mid_price_history_.pop_front();
        }
    }
    
    // Compute OFI if we have a previous snapshot
    if (snapshot_count_ > 0) {
        int prev_idx = (snapshot_index_ - 1 + SNAPSHOT_WINDOW) % SNAPSHOT_WINDOW;
        double ofi = compute_ofi(current, snapshot_buffer_[prev_idx]);
        
        // EMA update for OFI (replaces cumulative to prevent saturation)
        ofi_ema_ = OFI_EMA_ALPHA * ofi + (1.0 - OFI_EMA_ALPHA) * ofi_ema_;
        
        // Decay toward zero to prevent drift
        ofi_ema_ *= 0.99;
    }
    
    // Store snapshot in circular buffer
    snapshot_buffer_[snapshot_index_] = current;
    snapshot_index_ = (snapshot_index_ + 1) % SNAPSHOT_WINDOW;
    snapshot_count_ = std::min(snapshot_count_ + 1, SNAPSHOT_WINDOW);
    
    // Compute all signals
    compute_signals();
    
    // Return signals as vector (6 spread + 4 volume + 3 volatility = 13 signals)
    std::vector<double> retval;
    insert_signals(retval, *spread_signals_);
    insert_signals(retval, *volume_signals_);
    insert_signals(retval, *volatility_signals_);
    
    return retval;
}

BookSnapshot MarketSignalBuilder::compute_snapshot(const OrderBook& book) {
    BookSnapshot snapshot;
    
    const auto& bid_prices = book.bid_prices;
    const auto& ask_prices = book.ask_prices;
    const auto& bid_sizes = book.bid_sizes;
    const auto& ask_sizes = book.ask_sizes;
    
    // Store best bid/ask for OFI calculation
    snapshot.best_bid_price = bid_prices[0];
    snapshot.best_ask_price = ask_prices[0];
    
    // Mid price
    snapshot.mid_price = (bid_prices[0] + ask_prices[0]) * 0.5;
    
    // Market spread in basis points
    if (snapshot.mid_price > 0) {
        snapshot.market_spread_bps = (ask_prices[0] - bid_prices[0]) / snapshot.mid_price * 10000.0;
    }
    
    // Compute VWAP and total volumes over top 10 levels
    double bid_value = 0, ask_value = 0;
    double total_bid_vol = 0, total_ask_vol = 0;
    
    for (int i = 0; i < BOOK_DEPTH && i < static_cast<int>(bid_prices.size()); ++i) {
        bid_value += bid_prices[i] * bid_sizes[i];
        ask_value += ask_prices[i] * ask_sizes[i];
        total_bid_vol += bid_sizes[i];
        total_ask_vol += ask_sizes[i];
    }
    
    snapshot.total_bid_volume = total_bid_vol;
    snapshot.total_ask_volume = total_ask_vol;
    
    double vwap_bid = (total_bid_vol > 0) ? bid_value / total_bid_vol : bid_prices[0];
    double vwap_ask = (total_ask_vol > 0) ? ask_value / total_ask_vol : ask_prices[0];
    
    // Volume imbalance over 10 levels [-1, 1]
    double total_vol = total_bid_vol + total_ask_vol;
    if (total_vol > 0) {
        snapshot.volume_imbalance = (total_bid_vol - total_ask_vol) / total_vol;
    }
    
    // Book depth in basis points (always positive - distance from mid to VWAP)
    if (snapshot.mid_price > 0) {
        snapshot.bid_depth_bps = (snapshot.mid_price - vwap_bid) / snapshot.mid_price * 10000.0;
        snapshot.ask_depth_bps = (vwap_ask - snapshot.mid_price) / snapshot.mid_price * 10000.0;
        // Ensure non-negative
        snapshot.bid_depth_bps = std::max(0.0, snapshot.bid_depth_bps);
        snapshot.ask_depth_bps = std::max(0.0, snapshot.ask_depth_bps);
    }
    
    return snapshot;
}

double MarketSignalBuilder::compute_ofi(const BookSnapshot& current, const BookSnapshot& previous) {
    // Price-aware Order Flow Imbalance (academic OFI formula)
    // Bid side: if price went up, add current volume; if price went down, subtract previous volume
    // Ask side: if price went down, add current volume; if price went up, subtract previous volume
    
    double bid_flow = 0;
    if (current.best_bid_price > previous.best_bid_price) {
        bid_flow = current.total_bid_volume;
    } else if (current.best_bid_price < previous.best_bid_price) {
        bid_flow = -previous.total_bid_volume;
    } else {
        // Price unchanged: use volume difference
        bid_flow = current.total_bid_volume - previous.total_bid_volume;
    }
    
    double ask_flow = 0;
    if (current.best_ask_price < previous.best_ask_price) {
        ask_flow = current.total_ask_volume;
    } else if (current.best_ask_price > previous.best_ask_price) {
        ask_flow = -previous.total_ask_volume;
    } else {
        // Price unchanged: use volume difference
        ask_flow = current.total_ask_volume - previous.total_ask_volume;
    }
    
    // OFI = bid_flow - ask_flow, normalized to [-1, 1]
    double normalizer = (current.total_bid_volume + current.total_ask_volume + 
                         previous.total_bid_volume + previous.total_ask_volume) * 0.5;
    
    if (normalizer > 0) {
        double raw_ofi = (bid_flow - ask_flow) / normalizer;
        return std::clamp(raw_ofi, -1.0, 1.0);
    }
    return 0;
}

void MarketSignalBuilder::compute_signals() {
    compute_spread_signals();
    compute_volume_signals();
    compute_volatility_signals();
}

void MarketSignalBuilder::compute_spread_signals() {
    int curr_idx = (snapshot_index_ - 1 + SNAPSHOT_WINDOW) % SNAPSHOT_WINDOW;
    const BookSnapshot& current = snapshot_buffer_[curr_idx];
    
    // === Market spread signal ===
    // Typical crypto spreads: 0.5-10 bps
    // Normalize: 0 bps → 0, 5 bps → 0.5, 10+ bps → ~1
    spread_signals_->market_spread = sigmoid_normalize(current.market_spread_bps, 5.0, 0.4);
    
    // === Depth signals ===
    // Typical depth: 5-50 bps from mid to VWAP
    // Normalize: 0 bps → 0, 20 bps → 0.5, 40+ bps → ~1
    spread_signals_->bid_depth = sigmoid_normalize(current.bid_depth_bps, 20.0, 0.1);
    spread_signals_->ask_depth = sigmoid_normalize(current.ask_depth_bps, 20.0, 0.1);
    
    // Depth imbalance: which side has more depth?
    double total_depth = current.bid_depth_bps + current.ask_depth_bps;
    if (total_depth > 0.001) {
        spread_signals_->depth_imbalance = (current.bid_depth_bps - current.ask_depth_bps) / total_depth;
    } else {
        spread_signals_->depth_imbalance = 0.0;
    }
    
    // === Dynamics over window ===
    if (snapshot_count_ >= SNAPSHOT_WINDOW) {
        // Get oldest snapshot
        int oldest_idx = snapshot_index_;  // Next write position = oldest entry
        const BookSnapshot& oldest = snapshot_buffer_[oldest_idx];
        
        // Spread change: widening (+) or tightening (-)
        // Normalize change by typical spread magnitude
        double spread_diff = current.market_spread_bps - oldest.market_spread_bps;
        // Typical change: -5 to +5 bps over 500ms
        spread_signals_->spread_change = std::tanh(spread_diff / 3.0);
        
        // Depth change: increasing (+) or decreasing (-)
        double old_total_depth = oldest.bid_depth_bps + oldest.ask_depth_bps;
        double new_total_depth = current.bid_depth_bps + current.ask_depth_bps;
        double depth_diff = new_total_depth - old_total_depth;
        // Typical change: -10 to +10 bps over 500ms
        spread_signals_->depth_change = std::tanh(depth_diff / 10.0);
    } else {
        spread_signals_->spread_change = 0.0;
        spread_signals_->depth_change = 0.0;
    }
}

void MarketSignalBuilder::compute_volume_signals() {
    int curr_idx = (snapshot_index_ - 1 + SNAPSHOT_WINDOW) % SNAPSHOT_WINDOW;
    const BookSnapshot& current = snapshot_buffer_[curr_idx];
    
    // Current imbalance - already bounded [-1, 1] by construction
    volume_signals_->volume_imbalance = current.volume_imbalance;
    
    // OFI: EMA-smoothed, already bounded [-1, 1]
    volume_signals_->ofi = std::clamp(ofi_ema_, -1.0, 1.0);
    
    // Imbalance statistics over window
    if (snapshot_count_ >= SNAPSHOT_WINDOW) {
        double sum_imb = 0;
        for (int i = 0; i < SNAPSHOT_WINDOW; ++i) {
            sum_imb += snapshot_buffer_[i].volume_imbalance;
        }
        // Mean is bounded [-1, 1] since individual values are
        volume_signals_->volume_imbalance_mean = sum_imb / SNAPSHOT_WINDOW;
        
        // Trend: compare newest vs oldest
        int oldest_idx = snapshot_index_;
        const BookSnapshot& oldest = snapshot_buffer_[oldest_idx];
        // Raw trend is [-2, 2], divide by 2 to get [-1, 1]
        double raw_trend = current.volume_imbalance - oldest.volume_imbalance;
        volume_signals_->volume_imbalance_trend = raw_trend / 2.0;
    } else {
        volume_signals_->volume_imbalance_mean = current.volume_imbalance;
        volume_signals_->volume_imbalance_trend = 0.0;
    }
}

double MarketSignalBuilder::compute_volatility(int window) {
    // Compute realized volatility (std dev of returns) over window
    int n = static_cast<int>(mid_price_history_.size());
    if (n < window + 1) {
        return 0.0;  // Not enough data
    }
    
    // Use most recent 'window' returns
    int start = n - window - 1;
    double sum_ret = 0, sum_ret_sq = 0;
    int count = 0;
    
    for (int i = start; i < n - 1; ++i) {
        double prev = mid_price_history_[i];
        double curr = mid_price_history_[i + 1];
        if (prev > 0) {
            double ret = (curr - prev) / prev;  // Simple return
            sum_ret += ret;
            sum_ret_sq += ret * ret;
            count++;
        }
    }
    
    if (count < 2) return 0.0;
    
    double mean_ret = sum_ret / count;
    double variance = (sum_ret_sq / count) - (mean_ret * mean_ret);
    return std::sqrt(std::max(0.0, variance));
}

void MarketSignalBuilder::compute_volatility_signals() {
    // Short-term volatility (~1 second)
    double vol_short_raw = compute_volatility(VOL_SHORT_WINDOW);
    
    // Long-term volatility (~1 minute)
    double vol_long_raw = compute_volatility(VOL_LONG_WINDOW);
    
    // Update volatility baseline EMA (for adaptive normalization)
    if (vol_long_raw > 0) {
        vol_baseline_ema_ = VOL_BASELINE_ALPHA * vol_long_raw + 
                           (1.0 - VOL_BASELINE_ALPHA) * vol_baseline_ema_;
    }
    // Ensure minimum baseline
    double baseline = std::max(vol_baseline_ema_, 0.0001);
    
    // === Normalize volatilities to [0, 1] ===
    // Use sigmoid with adaptive baseline
    // Typical BTC volatility: 0.0001-0.001 per 100ms tick
    // midpoint = baseline (current "normal" level)
    // steepness chosen so that 2x baseline → ~0.75, 0.5x baseline → ~0.25
    
    // Short-term vol: compare to baseline
    volatility_signals_->vol_short = sigmoid_normalize(vol_short_raw, baseline, 1.0 / baseline);
    
    // Long-term vol: also normalized
    volatility_signals_->vol_long = sigmoid_normalize(vol_long_raw, baseline, 1.0 / baseline);
    
    // === Volatility regime signal [-1, 1] ===
    // -1 = calm (short << long), 0 = normal (short ≈ long), +1 = volatile (short >> long)
    if (vol_long_raw > 0.000001) {
        double ratio = vol_short_raw / vol_long_raw;
        // Map ratio to [-1, 1]:
        // ratio = 0.5 → -0.5 (calm)
        // ratio = 1.0 → 0 (normal)
        // ratio = 2.0 → +0.5 (elevated)
        // ratio = 4.0 → ~+1 (spike)
        // Use log ratio for symmetric treatment of high/low ratios
        double log_ratio = std::log(std::max(ratio, 0.01));  // Prevent log(0)
        // log_ratio: 0.5 → -0.69, 1.0 → 0, 2.0 → 0.69, 4.0 → 1.39
        volatility_signals_->vol_regime = std::tanh(log_ratio);
    } else if (vol_short_raw > 0.000001) {
        // Long-term is zero but short-term is non-zero: spike
        volatility_signals_->vol_regime = 1.0;
    } else {
        // Both near zero: neutral
        volatility_signals_->vol_regime = 0.0;
    }
}
