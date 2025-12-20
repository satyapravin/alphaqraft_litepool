// trade_signal_builder.h
#pragma once
#include "trade_reader.h"
#include <deque>

namespace RLTrader {
    struct TradeSignals {
        double buy_volume;          // buy_volume / (buy_volume + sell_volume), mapped to [-1, 1]
        double sell_volume;         // sell_volume / (buy_volume + sell_volume), mapped to [-1, 1]
        double volume_imbalance;    // (buy - sell) / (buy + sell), normalized [-1, 1]
        double trade_intensity;     // Volume per 100ms period (activity rate), normalized [-1, 1]
        double avg_volume_per_period; // Average volume per 100ms period, normalized [-1, 1]
        double price_impact;        // Price change per unit volume, normalized [-1, 1]
        double buy_pressure;        // buy_ema / (buy_ema + sell_ema), mapped to [-1, 1]
        double sell_pressure;       // sell_ema / (buy_ema + sell_ema), mapped to [-1, 1]
        double time_since_last_trade; // Normalized time since last trade [-1, 1]
        // Note: Trades are aggregated every 100ms. Each period is either an uptick (buy) or downtick (sell).
        // If a period has no trades, it simply doesn't appear in the feed.
        // Removed: trade_rate (not meaningful for fixed 100ms intervals)
        // Removed: avg_volume_per_period (redundant with trade_intensity)
    };

    class TradeSignalBuilder {
    private:
        // Rolling window for trade statistics (100ms aggregated periods)
        std::deque<Trade> recent_trades_;
        static constexpr int TRADE_WINDOW_SIZE = 100;  // Keep last 100 periods
        static constexpr long long TIME_WINDOW_US = 10000000;  // 10 seconds (100 periods at 100ms each)
        
        // EMA for momentum signals (adjusted for 100ms periodicity)
        double buy_volume_ema_ = 0.0;
        double sell_volume_ema_ = 0.0;
        static constexpr double EMA_ALPHA = 0.2;  // Higher alpha for 100ms aggregates (~5 period half-life)
        
        // Price impact tracking
        double prev_mid_price_ = 0.0;
        
        // Temporal tracking
        long long last_trade_timestamp_ = 0;  // For time_since_last_trade signal
        
        // Normalization parameters
        double max_volume_per_period_ = 1.0;  // Adaptive normalization for volume per 100ms
        double max_total_volume_ = 100.0;  // Max total volume in window (for intensity)
        static constexpr long long MAX_TIME_SINCE_TRADE_US = 10000000;  // 10 seconds max
        
    public:
        TradeSignalBuilder();
        
        // Process new trades and return signals
        TradeSignals add_trades(const std::vector<Trade>& trades, double mid_price, long long current_book_timestamp = 0);
        
        void reset();
    };
}

