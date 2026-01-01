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

#include <vector>
#include <array>
#include <deque>
#include <memory>
#include <cmath>
#include "orderbook.h"
#include "circ_table.h"

namespace RLTrader {

    // Number of orderbook levels to use for signal computation
    constexpr int BOOK_DEPTH = 10;
    // Number of snapshots to accumulate (10 snapshots = 1 sec at 100ms intervals)
    // Should match ticks_per_step for consistent step-level aggregation
    constexpr int SNAPSHOT_WINDOW = 10;
    // Volatility windows (at ~10 updates/sec)
    constexpr int VOL_SHORT_WINDOW = 10;    // ~1 second
    constexpr int VOL_LONG_WINDOW = 600;    // ~1 minute

    // Snapshot of key metrics from a single orderbook update
    struct BookSnapshot {
        double mid_price = 0;
        double market_spread_bps = 0;     // (ask - bid) / mid * 10000 (in basis points)
        double volume_imbalance = 0;      // (bid_vol - ask_vol) / (bid_vol + ask_vol) over 10 levels
        double total_bid_volume = 0;      // Total bid volume over 10 levels
        double total_ask_volume = 0;      // Total ask volume over 10 levels
        double best_bid_price = 0;        // Best bid price (for OFI calculation)
        double best_ask_price = 0;        // Best ask price (for OFI calculation)
        
        // Book depth: distance from mid to VWAP (always positive)
        double bid_depth_bps = 0;         // (mid - VWAP_bid) / mid * 10000
        double ask_depth_bps = 0;         // (VWAP_ask - mid) / mid * 10000
    } __attribute((packed));

    // Spread/depth signals - ALL BOUNDED TO [-1, 1]
    struct spread_signal_repository {
        // Market spread: how wide is the current bid-ask spread
        double market_spread = 0;             // [0, 1] normalized spread (tight=0, wide=1)
        
        // Book depth signals (both positive = more depth further from mid)
        double bid_depth = 0;                 // [0, 1] how deep is bid side
        double ask_depth = 0;                 // [0, 1] how deep is ask side
        double depth_imbalance = 0;           // [-1, 1] (bid-ask)/(bid+ask) depth
        
        // Dynamics over window
        double spread_change = 0;             // [-1, 1] spread widening (+) or tightening (-)
        double depth_change = 0;              // [-1, 1] depth increasing (+) or decreasing (-)
    } __attribute((packed));

    // Volume and order flow signals - ALL BOUNDED TO [-1, 1]
    struct volume_signal_repository {
        // Current snapshot signals (based on top 10 levels)
        double volume_imbalance = 0;          // [-1, 1] (bid-ask)/(bid+ask) over 10 levels
        
        // Accumulated signals over 5 snapshots (500ms window)
        double volume_imbalance_mean = 0;     // [-1, 1] Mean imbalance
        double volume_imbalance_trend = 0;    // [-1, 1] (newest - oldest) / 2
        
        // Order Flow Imbalance (EMA-smoothed)
        double ofi = 0;                       // [-1, 1] EMA of normalized OFI
    } __attribute((packed));

    // Volatility regime signals - ALL BOUNDED TO [-1, 1]
    struct volatility_signal_repository {
        double vol_short = 0;                 // [0, 1] short-term volatility (normalized)
        double vol_long = 0;                  // [0, 1] long-term volatility (normalized)
        double vol_regime = 0;                // [-1, 1] -1=calm, 0=normal, +1=volatile
    } __attribute((packed));

class MarketSignalBuilder {
public:
    explicit MarketSignalBuilder();

    // Process a new orderbook and return all signals as a vector
    std::vector<double> add_book(OrderBook& lob);

private:
    // Compute snapshot from current orderbook (using top 10 levels)
    BookSnapshot compute_snapshot(const OrderBook& book);
    
    // Compute OFI between current and previous snapshot
    double compute_ofi(const BookSnapshot& current, const BookSnapshot& previous);
    
    // Compute all signals from accumulated snapshots
    void compute_signals();
    
    // Compute spread/depth signals
    void compute_spread_signals();
    
    // Compute volume/imbalance signals
    void compute_volume_signals();
    
    // Compute volatility regime signals
    void compute_volatility_signals();
    
    // Helper: compute volatility from returns
    double compute_volatility(int window);
    
    // Helper: sigmoid normalization to [0, 1] with configurable midpoint
    static double sigmoid_normalize(double x, double midpoint, double steepness);

private:
    // Circular buffer to store last 5 snapshots
    std::array<BookSnapshot, SNAPSHOT_WINDOW> snapshot_buffer_;
    int snapshot_count_ = 0;
    int snapshot_index_ = 0;
    
    // EMA for OFI (replaces cumulative which could saturate)
    double ofi_ema_ = 0;
    static constexpr double OFI_EMA_ALPHA = 0.2;  // ~5 sample half-life
    
    // EMA for volatility baseline
    double vol_baseline_ema_ = 0;
    static constexpr double VOL_BASELINE_ALPHA = 0.01;  // ~100 sample half-life
    
    // Mid-price history for volatility (longer window)
    std::deque<double> mid_price_history_;
    
    // Output signal repositories
    std::unique_ptr<spread_signal_repository> spread_signals_;
    std::unique_ptr<volume_signal_repository> volume_signals_;
    std::unique_ptr<volatility_signal_repository> volatility_signals_;
};
}
