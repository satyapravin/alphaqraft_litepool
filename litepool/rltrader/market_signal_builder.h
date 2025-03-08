#pragma once

#include <vector>
#include <memory>
#include "orderbook.h"
#include "circ_table.h"

namespace RLTrader {
    struct volume_signal_repository {
        double volume_imbalance_signal_0 = 0;
        double volume_imbalance_signal_1 = 0;
        double volume_imbalance_signal_2 = 0;
        double volume_imbalance_signal_3 = 0;
        double volume_imbalance_signal_4 = 0;

        double volume_ofi_signal_0 = 0; // 100 msec level 1
        double volume_ofi_signal_1 = 0; // 100 msec level 5
        double volume_ofi_signal_2 = 0; // 1 sec level 1
        double volume_ofi_signal_3 = 0; // 1 sec level 5
        double volume_ofi_signal_4 = 0; // 2 sec level 1
        double volume_ofi_signal_5 = 0; // 2 sec level 5
        double volume_ofi_signal_6 = 0; // 3 sec level 1
        double volume_ofi_signal_7 = 0; // 3 sec level 5

    } __attribute((packed));

    struct spread_signal_repository {
        double norm_vwap_bid_spread_signal_0 = 0;
        double norm_vwap_bid_spread_signal_1 = 0;
        double norm_vwap_bid_spread_signal_2 = 0;
        double norm_vwap_bid_spread_signal_3 = 0;
        double norm_vwap_bid_spread_signal_4 = 0;

        double norm_vwap_ask_spread_signal_0 = 0;
        double norm_vwap_ask_spread_signal_1 = 0;
        double norm_vwap_ask_spread_signal_2 = 0;
        double norm_vwap_ask_spread_signal_3 = 0;
        double norm_vwap_ask_spread_signal_4 = 0;

        double micro_spread_signal_0 = 0;
        double micro_spread_signal_1 = 0;
        double micro_spread_signal_2 = 0;
        double micro_spread_signal_3 = 0;
        double micro_spread_signal_4 = 0;

        double bid_fill_spread_signal_0 = 0;
        double bid_fill_spread_signal_1 = 0;
        double bid_fill_spread_signal_2 = 0;
        double bid_fill_spread_signal_3 = 0;
        double bid_fill_spread_signal_4 = 0;

        double ask_fill_spread_signal_0 = 0;
        double ask_fill_spread_signal_1 = 0;
        double ask_fill_spread_signal_2 = 0;
        double ask_fill_spread_signal_3 = 0;
        double ask_fill_spread_signal_4 = 0;
    } __attribute((packed));

    struct price_signal_repository {
        double mid_price_signal = 0;
        double vwap_bid_price_signal_0 = 0;
        double vwap_bid_price_signal_1 = 0;
        double vwap_bid_price_signal_2 = 0;
        double vwap_bid_price_signal_3 = 0;
        double vwap_bid_price_signal_4 = 0;

        double vwap_ask_price_signal_0 = 0;
        double vwap_ask_price_signal_1 = 0;
        double vwap_ask_price_signal_2 = 0;
        double vwap_ask_price_signal_3 = 0;
        double vwap_ask_price_signal_4 = 0;

        double norm_vwap_bid_price_signal_0 = 0;
        double norm_vwap_bid_price_signal_1 = 0;
        double norm_vwap_bid_price_signal_2 = 0;
        double norm_vwap_bid_price_signal_3 = 0;
        double norm_vwap_bid_price_signal_4 = 0;

        double norm_vwap_ask_price_signal_0 = 0;
        double norm_vwap_ask_price_signal_1 = 0;
        double norm_vwap_ask_price_signal_2 = 0;
        double norm_vwap_ask_price_signal_3 = 0;
        double norm_vwap_ask_price_signal_4 = 0;

        double micro_price_signal_0 = 0;
        double micro_price_signal_1 = 0;
        double micro_price_signal_2 = 0;
        double micro_price_signal_3 = 0;
        double micro_price_signal_4 = 0;

        double bid_fill_price_signal_0 = 0;
        double bid_fill_price_signal_1 = 0;
        double bid_fill_price_signal_2 = 0;
        double bid_fill_price_signal_3 = 0;
        double bid_fill_price_signal_4 = 0;

        double ask_fill_price_signal_0 = 0;
        double ask_fill_price_signal_1 = 0;
        double ask_fill_price_signal_2 = 0;
        double ask_fill_price_signal_3 = 0;
        double ask_fill_price_signal_4 = 0;
    } __attribute((packed));

class MarketSignalBuilder {
public:
    explicit MarketSignalBuilder();

    std::vector<double> add_book(OrderBook& lob);


private:
    void compute_signals(const OrderBook& book);

    static double compute_ofi(double curr_bid_price, double curr_bid_size,
                       double curr_ask_price, double curr_ask_size,
                       double prev_bid_price, double prev_bid_size,
                       double prev_ask_price, double prev_ask_size);

    std::tuple<double, double>
    compute_lagged_ofi(const price_signal_repository& repo,
                                             const FixedVector<double, 20>& cum_bid_sizes,
                                             const FixedVector<double, 20>& cum_ask_sizes,
                                             int lag);

    void compute_price_signals( price_signal_repository& repo,
                                const FixedVector<double, 20>& current_bid_prices,
                                const FixedVector<double, 20>& current_ask_prices,
                                const FixedVector<double, 20>& current_bid_sizes,
                                const FixedVector<double, 20>& current_ask_sizes,
                                const FixedVector<double, 20>& cum_bid_sizes,
                                const FixedVector<double, 20>& cum_ask_sizes,
                                const FixedVector<double, 20>& cum_bid_amounts,
                                const FixedVector<double, 20>& cum_ask_amounts);

    void compute_ewma_signals(const price_signal_repository& repo);
    
    void compute_spread_signals(const price_signal_repository& repo) const;

    void compute_volume_signals(const FixedVector<double, 20>& bid_sizes,
                                const FixedVector<double, 20>& ask_sizes,
                                const FixedVector<double, 20>& cum_bid_sizes,
                                const FixedVector<double, 20>& cum_ask_sizes) const;

private:
    double ewma_30 = 2.0 / 16.0;
    double ewma_60 = 2.0 / 61.0;
    double ewma_120 = 2.0 / 121.0;
    double ewma_300 = 2.0 / 301.0;

    TemporalTable previous_bid_prices;
    TemporalTable previous_ask_prices;
    TemporalTable previous_bid_amounts;
    TemporalTable previous_ask_amounts;
    price_signal_repository previous_price_signal;
    price_signal_repository ewma_price_signal_30;
    price_signal_repository ewma_price_signal_60;
    price_signal_repository ewma_price_signal_120;
    price_signal_repository ewma_price_signal_300;
    std::unique_ptr<price_signal_repository> raw_price_diff_signals;
    std::unique_ptr<spread_signal_repository> raw_spread_signals;
    std::unique_ptr<volume_signal_repository> raw_volume_signals;
    std::unique_ptr<price_signal_repository> ewma_diff_signals_30;
    std::unique_ptr<price_signal_repository> ewma_diff_signals_60;
    std::unique_ptr<price_signal_repository> ewma_diff_signals_120;
    std::unique_ptr<price_signal_repository> ewma_diff_signals_300;
};
}
