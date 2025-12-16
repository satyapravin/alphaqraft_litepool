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
    // Number of snapshots to accumulate (5 snapshots = 500ms at 100ms intervals)
    constexpr int SNAPSHOT_WINDOW = 5;
    // Volatility windows (at ~10 updates/sec)
    constexpr int VOL_SHORT_WINDOW = 10;    // ~1 second
    constexpr int VOL_LONG_WINDOW = 600;    // ~1 minute

    // Snapshot of key metrics from a single orderbook update
    struct BookSnapshot {
        double mid_price = 0;
        double volume_imbalance = 0;      // (bid_vol - ask_vol) / (bid_vol + ask_vol) over 10 levels
        double total_bid_volume = 0;      // Total bid volume over 10 levels
        double total_ask_volume = 0;      // Total ask volume over 10 levels
        double best_bid_price = 0;        // Best bid price (for OFI calculation)
        double best_ask_price = 0;        // Best ask price (for OFI calculation)
        
        // Normalized spreads (stored for mean/volatility computation)
        double bid_spread = 0;            // (VWAP_bid - mid) / mid * 100
        double ask_spread = 0;            // (VWAP_ask - mid) / mid * 100
    } __attribute((packed));

    // Volume and order flow signals - ALL BOUNDED TO [-1, 1]
    struct volume_signal_repository {
        // Current snapshot signals (based on top 10 levels)
        double volume_imbalance = 0;          // [-1, 1] (bid-ask)/(bid+ask) over 10 levels
        
        // Accumulated signals over 5 snapshots (500ms window)
        double volume_imbalance_mean = 0;     // [-1, 1] Mean imbalance
        double volume_imbalance_trend = 0;    // [-1, 1] tanh(newest - oldest)
        
        // Order Flow Imbalance (EMA-smoothed, tanh-bounded)
        double ofi_cumulative = 0;            // [-1, 1] tanh(EMA of OFI)
    } __attribute((packed));

    // Spread signals for quote placement - ALL BOUNDED TO [-1, 1]
    struct spread_signal_repository {
        // Current snapshot (tanh-scaled)
        double bid_spread = 0;                // [-1, 1] tanh((VWAP_bid - mid) / mid * 1000)
        double ask_spread = 0;                // [-1, 1] tanh((VWAP_ask - mid) / mid * 1000)
        
        // Bid spread dynamics over 5 snapshots
        double bid_spread_mean = 0;           // [-1, 1] tanh(mean * scale)
        double bid_spread_volatility = 0;     // [-1, 1] tanh(CV), CV = σ/|μ|
        
        // Ask spread dynamics over 5 snapshots
        double ask_spread_mean = 0;           // [-1, 1] tanh(mean * scale)
        double ask_spread_volatility = 0;     // [-1, 1] tanh(CV), CV = σ/|μ|
    } __attribute((packed));

    // Volatility regime signals - ALL BOUNDED TO [-1, 1]
    struct volatility_signal_repository {
        double vol_short = 0;                 // [-1, 1] tanh(vol_1s * scale), short-term volatility
        double vol_long = 0;                  // [-1, 1] tanh(vol_1m * scale), long-term volatility
        double vol_ratio = 0;                 // [-1, 1] tanh(short/long - 1), spike detector
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
    
    // Compute spread signals
    void compute_spread_signals();
    
    // Compute volume/imbalance signals
    void compute_volume_signals();
    
    // Compute volatility regime signals
    void compute_volatility_signals();
    
    // Helper: compute volatility from returns
    double compute_volatility(int window);

private:
    // Circular buffer to store last 5 snapshots
    std::array<BookSnapshot, SNAPSHOT_WINDOW> snapshot_buffer_;
    int snapshot_count_ = 0;
    int snapshot_index_ = 0;
    
    // Cumulative OFI over window
    double cumulative_ofi_ = 0;
    
    // Mid-price history for volatility (longer window)
    std::deque<double> mid_price_history_;
    
    // Output signal repositories
    std::unique_ptr<spread_signal_repository> spread_signals_;
    std::unique_ptr<volume_signal_repository> volume_signals_;
    std::unique_ptr<volatility_signal_repository> volatility_signals_;
};
}
