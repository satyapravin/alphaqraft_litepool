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
      cumulative_ofi_(0),
      mid_price_history_(),
      spread_signals_(std::make_unique<spread_signal_repository>()),
      volume_signals_(std::make_unique<volume_signal_repository>()),
      volatility_signals_(std::make_unique<volatility_signal_repository>())
{
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
        
        // EMA-style cumulative OFI (decay factor ~0.8 gives ~5 sample half-life)
        constexpr double OFI_DECAY = 0.8;
        cumulative_ofi_ = OFI_DECAY * cumulative_ofi_ + ofi;
    }
    
    // Store snapshot in circular buffer
    snapshot_buffer_[snapshot_index_] = current;
    snapshot_index_ = (snapshot_index_ + 1) % SNAPSHOT_WINDOW;
    snapshot_count_ = std::min(snapshot_count_ + 1, SNAPSHOT_WINDOW);
    
    // Compute all signals
    compute_signals();
    
    // Return signals as vector (10 spread/volume + 3 volatility = 13 signals)
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
    
    // Compute VWAP and total volumes over top 10 levels
    double bid_amount = 0, ask_amount = 0;
    double total_bid_vol = 0, total_ask_vol = 0;
    
    for (int i = 0; i < BOOK_DEPTH && i < static_cast<int>(bid_prices.size()); ++i) {
        bid_amount += bid_prices[i] * bid_sizes[i];
        ask_amount += ask_prices[i] * ask_sizes[i];
        total_bid_vol += bid_sizes[i];
        total_ask_vol += ask_sizes[i];
    }
    
    snapshot.total_bid_volume = total_bid_vol;
    snapshot.total_ask_volume = total_ask_vol;
    
    double vwap_bid = (total_bid_vol > 0) ? bid_amount / total_bid_vol : bid_prices[0];
    double vwap_ask = (total_ask_vol > 0) ? ask_amount / total_ask_vol : ask_prices[0];
    
    // Volume imbalance over 10 levels
    double total_vol = total_bid_vol + total_ask_vol;
    if (total_vol > 0) {
        snapshot.volume_imbalance = (total_bid_vol - total_ask_vol) / total_vol;
    }
    
    // Compute normalized spreads (stored for mean/volatility computation)
    if (snapshot.mid_price > 0) {
        snapshot.bid_spread = (vwap_bid - snapshot.mid_price) / snapshot.mid_price * 100.0;
        snapshot.ask_spread = (vwap_ask - snapshot.mid_price) / snapshot.mid_price * 100.0;
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
    
    // OFI = bid_flow - ask_flow, normalized
    double normalizer = (current.total_bid_volume + current.total_ask_volume + 
                         previous.total_bid_volume + previous.total_ask_volume) * 0.5;
    
    if (normalizer > 0) {
        return (bid_flow - ask_flow) / normalizer;
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
    
    // Spread scaling: typical crypto perpetual spreads are 0.001-0.05% after *100
    // Scale by 100 to get tanh input in useful range:
    //   0.005% (tight) → tanh(0.5) ≈ 0.46
    //   0.02% (normal) → tanh(2.0) ≈ 0.96
    //   0.1% (wide)    → tanh(10)  ≈ 1.0
    constexpr double SPREAD_SCALE = 100.0;
    
    // Current snapshot spreads (bounded to [-1, 1])
    spread_signals_->bid_spread = std::tanh(current.bid_spread * SPREAD_SCALE);
    spread_signals_->ask_spread = std::tanh(current.ask_spread * SPREAD_SCALE);
    
    // Spread statistics over window
    if (snapshot_count_ >= SNAPSHOT_WINDOW) {
        double sum_bid = 0, sum_bid_sq = 0;
        double sum_ask = 0, sum_ask_sq = 0;
        
        for (int i = 0; i < SNAPSHOT_WINDOW; ++i) {
            const BookSnapshot& snap = snapshot_buffer_[i];
            sum_bid += snap.bid_spread;
            sum_bid_sq += snap.bid_spread * snap.bid_spread;
            sum_ask += snap.ask_spread;
            sum_ask_sq += snap.ask_spread * snap.ask_spread;
        }
        
        // Bid spread mean and volatility
        double mean_bid = sum_bid / SNAPSHOT_WINDOW;
        double var_bid = sum_bid_sq / SNAPSHOT_WINDOW - mean_bid * mean_bid;
        // Bound mean to [-1, 1]
        spread_signals_->bid_spread_mean = std::tanh(mean_bid * SPREAD_SCALE);
        
        // Coefficient of Variation for volatility
        constexpr double MIN_SPREAD_FOR_CV = 0.01;
        double raw_bid_vol = std::sqrt(std::max(0.0, var_bid));
        double cv_bid = raw_bid_vol / std::max(std::abs(mean_bid), MIN_SPREAD_FOR_CV);
        spread_signals_->bid_spread_volatility = std::tanh(cv_bid);
        
        // Ask spread mean and volatility
        double mean_ask = sum_ask / SNAPSHOT_WINDOW;
        double var_ask = sum_ask_sq / SNAPSHOT_WINDOW - mean_ask * mean_ask;
        // Bound mean to [-1, 1]
        spread_signals_->ask_spread_mean = std::tanh(mean_ask * SPREAD_SCALE);
        
        double raw_ask_vol = std::sqrt(std::max(0.0, var_ask));
        double cv_ask = raw_ask_vol / std::max(std::abs(mean_ask), MIN_SPREAD_FOR_CV);
        spread_signals_->ask_spread_volatility = std::tanh(cv_ask);
    }
}

void MarketSignalBuilder::compute_volume_signals() {
    int curr_idx = (snapshot_index_ - 1 + SNAPSHOT_WINDOW) % SNAPSHOT_WINDOW;
    const BookSnapshot& current = snapshot_buffer_[curr_idx];
    
    // Current imbalance - already bounded [-1, 1] by construction
    volume_signals_->volume_imbalance = current.volume_imbalance;
    
    // Cumulative OFI - bound to [-1, 1] using tanh
    volume_signals_->ofi_cumulative = std::tanh(cumulative_ofi_);
    
    // Imbalance statistics over window
    if (snapshot_count_ >= SNAPSHOT_WINDOW) {
        double sum_imb = 0;
        for (int i = 0; i < SNAPSHOT_WINDOW; ++i) {
            sum_imb += snapshot_buffer_[i].volume_imbalance;
        }
        // Mean is bounded [-1, 1] since individual values are
        volume_signals_->volume_imbalance_mean = sum_imb / SNAPSHOT_WINDOW;
        
        // Trend: compare newest vs oldest, bound to [-1, 1]
        // Raw trend could be [-2, 2], scale by 0.5 and tanh for strict bound
        int oldest_idx = snapshot_index_;
        const BookSnapshot& oldest = snapshot_buffer_[oldest_idx];
        double raw_trend = current.volume_imbalance - oldest.volume_imbalance;
        volume_signals_->volume_imbalance_trend = std::tanh(raw_trend);
    }
}

double MarketSignalBuilder::compute_volatility(int window) {
    // Compute realized volatility (std dev of log returns) over window
    int n = static_cast<int>(mid_price_history_.size());
    if (n < window + 1) {
        return 0.0;  // Not enough data
    }
    
    // Use most recent 'window' returns
    int start = n - window - 1;
    double sum_ret = 0, sum_ret_sq = 0;
    
    for (int i = start; i < n - 1; ++i) {
        double prev = mid_price_history_[i];
        double curr = mid_price_history_[i + 1];
        if (prev > 0) {
            double ret = (curr - prev) / prev;  // Simple return
            sum_ret += ret;
            sum_ret_sq += ret * ret;
        }
    }
    
    double mean_ret = sum_ret / window;
    double variance = (sum_ret_sq / window) - (mean_ret * mean_ret);
    return std::sqrt(std::max(0.0, variance));
}

void MarketSignalBuilder::compute_volatility_signals() {
    // Volatility scaling: typical BTC volatility is 0.0001-0.001 per 100ms tick
    // Scale by 10000 to get tanh input in useful range:
    //   0.0001 (calm)    → tanh(1.0)  ≈ 0.76
    //   0.0005 (normal)  → tanh(5.0)  ≈ 0.99
    //   0.001  (volatile)→ tanh(10)   ≈ 1.0
    constexpr double VOL_SCALE = 10000.0;
    
    // Short-term volatility (~1 second)
    double vol_short_raw = compute_volatility(VOL_SHORT_WINDOW);
    volatility_signals_->vol_short = std::tanh(vol_short_raw * VOL_SCALE);
    
    // Long-term volatility (~1 minute)
    double vol_long_raw = compute_volatility(VOL_LONG_WINDOW);
    volatility_signals_->vol_long = std::tanh(vol_long_raw * VOL_SCALE);
    
    // Volatility ratio (spike detector)
    // If short > long, we're in a spike; if short < long, we're calming down
    // Ratio - 1 gives us deviation from "normal": 0 = equal, >0 = spike, <0 = calm
    if (vol_long_raw > 0.000001) {
        double ratio = vol_short_raw / vol_long_raw;
        volatility_signals_->vol_ratio = std::tanh(ratio - 1.0);
    } else if (vol_short_raw > 0.000001) {
        // Long-term is zero but short-term is non-zero: spike
        volatility_signals_->vol_ratio = 1.0;
    } else {
        // Both near zero
        volatility_signals_->vol_ratio = 0.0;
    }
}
