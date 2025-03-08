#include "market_signal_builder.h"
#include <numeric>
#include <map>
#include <cmath>
#include "orderbook.h"
#include "rl_macros.h"

using namespace RLTrader;

MarketSignalBuilder::MarketSignalBuilder()
              :previous_bid_prices(30),
               previous_ask_prices(30),
               previous_bid_amounts(30),
               previous_ask_amounts(30),
               previous_price_signal{},
               ewma_price_signal_30{},
               ewma_price_signal_60{},
               ewma_price_signal_120{},
               ewma_price_signal_300{},
               raw_price_diff_signals(std::make_unique<price_signal_repository>()),                    
               raw_spread_signals(std::make_unique<spread_signal_repository>()),                        
               raw_volume_signals(std::make_unique<volume_signal_repository>()),                         
               ewma_diff_signals_30(std::make_unique<price_signal_repository>()),                         
               ewma_diff_signals_60(std::make_unique<price_signal_repository>()),                         
               ewma_diff_signals_120(std::make_unique<price_signal_repository>()),                         
               ewma_diff_signals_300(std::make_unique<price_signal_repository>())
{
    
}


void cumulative_prod_sum(const FixedVector<double, 20>& first, const FixedVector<double, 20>& second,
                                        FixedVector<double, 20>& result) {
    double cum_product_sum = 0;

    for(auto ii=0; ii < first.size(); ++ii) {
        cum_product_sum += first[ii] * second[ii];
        result[ii] = cum_product_sum;
    }
}

void get_cumulative_sizes(const FixedVector<double, 20>& sizes, FixedVector<double, 20>& retval) {
    std::partial_sum(sizes.begin(), sizes.end(), retval.begin());
}

double fill_price(const FixedVector<double, 20>& bid_prices, const FixedVector<double, 20>& bid_sizes, double size) {
    double fill_amount = 0;
    double fill_size = size;
    for(auto ii = 0; ii < bid_prices.size(); ++ii) {
        if (fill_size <= 0) break;
        fill_amount += std::min(bid_sizes[ii], fill_size) * bid_prices[ii];
        fill_size -= bid_sizes[ii];
    }

    if (fill_size > 0) {
        size -= fill_size;
    }

    return fill_amount / size;
}

double micro_price(const double& bid_price, const double& ask_price,
                   const double& bid_size, const double& ask_size) {
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size);
}

std::vector<double> MarketSignalBuilder::add_book(OrderBook& book) {
    compute_signals(book);

    std::vector<double> retval;
    insert_signals(retval, *raw_price_diff_signals);
    insert_signals(retval, *raw_spread_signals);
    insert_signals(retval, *raw_volume_signals);
    insert_signals(retval, *ewma_diff_signals_30);
    insert_signals(retval, *ewma_diff_signals_60);
    insert_signals(retval, *ewma_diff_signals_120);
    insert_signals(retval, *ewma_diff_signals_300);

    previous_bid_prices.addRow(book.bid_prices);
    previous_ask_prices.addRow(book.ask_prices);
    previous_bid_amounts.addRow(book.bid_sizes);
    previous_ask_amounts.addRow(book.ask_sizes);
    return retval;
}

void MarketSignalBuilder::compute_signals(const OrderBook& book) {
    auto current_bid_prices = book.bid_prices;
    auto current_ask_prices = book.ask_prices;
    auto current_bid_sizes = book.bid_sizes;
    auto current_ask_sizes = book.ask_sizes;

    FixedVector<double, 20> cum_bid_sizes;
    FixedVector<double, 20> cum_ask_sizes;
    get_cumulative_sizes(current_bid_sizes, cum_bid_sizes);
    get_cumulative_sizes(current_ask_sizes, cum_ask_sizes);

    FixedVector<double, 20> cum_bid_amounts;
    FixedVector<double, 20> cum_ask_amounts;
    cumulative_prod_sum(current_bid_prices, current_bid_sizes, cum_bid_amounts);
    cumulative_prod_sum(current_ask_prices, current_ask_sizes, cum_ask_amounts);

    price_signal_repository repo;

    compute_price_signals(repo,
                          current_bid_prices,
                          current_ask_prices,
                          current_bid_sizes,
                          current_ask_sizes,
                          cum_bid_sizes,
                          cum_ask_sizes,
                          cum_bid_amounts,
                          cum_ask_amounts);

    compute_ewma_signals(repo);

    compute_spread_signals(repo);

    compute_volume_signals(current_bid_sizes, current_ask_sizes,
                           cum_bid_sizes, cum_ask_sizes);

    // here we compute OFI
    auto ofis = compute_lagged_ofi(repo, cum_bid_sizes, cum_ask_sizes, 1);
    raw_volume_signals->volume_ofi_signal_0 = std::get<0>(ofis);
    raw_volume_signals->volume_ofi_signal_1 = std::get<1>(ofis);

    ofis = compute_lagged_ofi(repo, cum_bid_sizes, cum_ask_sizes, 9);
    raw_volume_signals->volume_ofi_signal_2 = std::get<0>(ofis);
    raw_volume_signals->volume_ofi_signal_3 = std::get<1>(ofis);

    ofis = compute_lagged_ofi(repo, cum_bid_sizes, cum_ask_sizes, 19);
    raw_volume_signals->volume_ofi_signal_4 = std::get<0>(ofis);
    raw_volume_signals->volume_ofi_signal_5 = std::get<1>(ofis);

    ofis = compute_lagged_ofi(repo, cum_bid_sizes, cum_ask_sizes, 29);
    raw_volume_signals->volume_ofi_signal_6 = std::get<0>(ofis);
    raw_volume_signals->volume_ofi_signal_7 = std::get<1>(ofis);
}

std::tuple<double, double>
MarketSignalBuilder::compute_lagged_ofi(const price_signal_repository& repo,
                                        const FixedVector<double, 20>& cum_bid_sizes,
                                        const FixedVector<double, 20>& cum_ask_sizes,
                                        const int lag)
{
    auto current_bid_price = repo.vwap_bid_price_signal_0;
    auto current_ask_price = repo.vwap_ask_price_signal_0;
    auto current_bid_price_5 = repo.vwap_bid_price_signal_4;
    auto current_ask_price_5 = repo.vwap_ask_price_signal_4;
    auto current_bid_size = cum_bid_sizes[0];
    auto current_ask_size = cum_ask_sizes[0];
    auto current_bid_size_5 = cum_bid_sizes[5];
    auto current_ask_size_5 = cum_ask_sizes[5];

    auto lagged_bid_prices = previous_bid_prices.get(previous_bid_prices.get_lagged_row(lag));
    auto lagged_ask_prices = previous_ask_prices.get(previous_ask_prices.get_lagged_row(lag));
    auto lagged_bid_sizes = previous_bid_amounts.get(previous_bid_amounts.get_lagged_row(lag));
    auto lagged_ask_sizes = previous_ask_amounts.get(previous_ask_amounts.get_lagged_row(lag));

    FixedVector<double, 20> previous_cum_bid_sizes;
    FixedVector<double, 20> previous_cum_ask_sizes;
    FixedVector<double, 20> previous_cum_bid_amounts;
    FixedVector<double, 20> previous_cum_ask_amounts;
    get_cumulative_sizes(lagged_bid_sizes, previous_cum_bid_sizes);
    get_cumulative_sizes(lagged_ask_sizes, previous_cum_ask_sizes);

    cumulative_prod_sum(lagged_bid_prices, lagged_bid_sizes, previous_cum_bid_amounts);
    cumulative_prod_sum(lagged_ask_prices, lagged_ask_sizes, previous_cum_ask_amounts);

    double previous_vwap_bid_price_5 = previous_cum_bid_amounts[4] / previous_cum_bid_sizes[4];
    double previous_vwap_ask_price_5 = previous_cum_ask_amounts[4] / previous_cum_ask_sizes[4];

    double level_1 = compute_ofi(current_bid_price, current_bid_size,
                                 current_ask_price, current_ask_size,
                                 lagged_bid_prices[0], lagged_bid_sizes[0],
                                 lagged_ask_prices[0], lagged_ask_sizes[0]);

    double level_5 = compute_ofi(current_bid_price_5, current_bid_size_5,
                                 current_ask_price_5, current_ask_size_5,
                                 previous_vwap_bid_price_5, previous_cum_bid_sizes[4],
                                 previous_vwap_ask_price_5, previous_cum_ask_sizes[4]);
    return {level_1, level_5};
}

double MarketSignalBuilder::compute_ofi(double curr_bid_price, double curr_bid_size,
                                      double curr_ask_price, double curr_ask_size,
                                      double prev_bid_price, double prev_bid_size,
                                      double prev_ask_price, double prev_ask_size) {
    auto retval =  (curr_bid_price >= prev_bid_price ? curr_bid_size : 0) -
                         (curr_bid_price <= prev_bid_price ? prev_bid_size : 0) -
                         (curr_ask_price <= prev_ask_price ? curr_ask_size : 0) +
                         (curr_ask_price >= prev_ask_price ? prev_ask_size : 0);

    double normalizer = (curr_bid_size + curr_ask_size + prev_bid_size + prev_ask_size) * 0.5;
    return retval / normalizer;
}

void MarketSignalBuilder::compute_volume_signals(const FixedVector<double, 20>& bid_sizes,
                                                 const FixedVector<double, 20>& ask_sizes,
                                                 const FixedVector<double, 20>& cum_bid_sizes,
                                                 const FixedVector<double, 20>& cum_ask_sizes) const {
    raw_volume_signals->volume_imbalance_signal_0 = (bid_sizes[0] - ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0]);
    raw_volume_signals->volume_imbalance_signal_1 = (cum_bid_sizes[1] - cum_ask_sizes[1]) / (cum_bid_sizes[1] + cum_ask_sizes[1]);
    raw_volume_signals->volume_imbalance_signal_2 = (cum_bid_sizes[2] - cum_ask_sizes[2]) / (cum_bid_sizes[2] + cum_ask_sizes[2]);
    raw_volume_signals->volume_imbalance_signal_3 = (cum_bid_sizes[3] - cum_ask_sizes[3]) / (cum_bid_sizes[3] + cum_ask_sizes[3]);
    raw_volume_signals->volume_imbalance_signal_4 = (cum_bid_sizes[4] - cum_ask_sizes[4]) / (cum_bid_sizes[4] + cum_ask_sizes[4]);
}

void MarketSignalBuilder::compute_spread_signals(const price_signal_repository& repo) const {
    raw_spread_signals->norm_vwap_bid_spread_signal_0 = (repo.norm_vwap_bid_price_signal_0 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_bid_spread_signal_1 = (repo.norm_vwap_bid_price_signal_1 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_bid_spread_signal_2 = (repo.norm_vwap_bid_price_signal_2 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_bid_spread_signal_3 = (repo.norm_vwap_bid_price_signal_3 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_bid_spread_signal_4 = (repo.norm_vwap_bid_price_signal_4 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;

    raw_spread_signals->norm_vwap_ask_spread_signal_0 = (repo.norm_vwap_ask_price_signal_0 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_ask_spread_signal_1 = (repo.norm_vwap_ask_price_signal_1 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_ask_spread_signal_2 = (repo.norm_vwap_ask_price_signal_2 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_ask_spread_signal_3 = (repo.norm_vwap_ask_price_signal_3 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->norm_vwap_ask_spread_signal_4 = (repo.norm_vwap_ask_price_signal_4 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;

    raw_spread_signals->micro_spread_signal_0 = (repo.micro_price_signal_0 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->micro_spread_signal_1 = (repo.micro_price_signal_1 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->micro_spread_signal_2 = (repo.micro_price_signal_2 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->micro_spread_signal_3 = (repo.micro_price_signal_3 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->micro_spread_signal_4 = (repo.micro_price_signal_4 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;

    raw_spread_signals->bid_fill_spread_signal_0 = (repo.bid_fill_price_signal_0 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->bid_fill_spread_signal_1 = (repo.bid_fill_price_signal_1 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->bid_fill_spread_signal_2 = (repo.bid_fill_price_signal_2 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->bid_fill_spread_signal_3 = (repo.bid_fill_price_signal_3 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->bid_fill_spread_signal_4 = (repo.bid_fill_price_signal_4 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;

    raw_spread_signals->ask_fill_spread_signal_0 = (repo.ask_fill_price_signal_0 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->ask_fill_spread_signal_1 = (repo.ask_fill_price_signal_1 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->ask_fill_spread_signal_2 = (repo.ask_fill_price_signal_2 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->ask_fill_spread_signal_3 = (repo.ask_fill_price_signal_3 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
    raw_spread_signals->ask_fill_spread_signal_4 = (repo.ask_fill_price_signal_4 - repo.mid_price_signal) * 100.0 / repo.mid_price_signal;
}

void MarketSignalBuilder::compute_price_signals(price_signal_repository& raw_price_repo,
                                          const FixedVector<double, 20>& current_bid_prices,
                                          const FixedVector<double, 20>& current_ask_prices,
                                          const FixedVector<double, 20>& current_bid_sizes,
                                          const FixedVector<double, 20>& current_ask_sizes,
                                          const FixedVector<double, 20>& cum_bid_sizes,
                                          const FixedVector<double, 20>& cum_ask_sizes,
                                          const FixedVector<double, 20>& cum_bid_amounts,
                                          const FixedVector<double, 20>& cum_ask_amounts) {
    raw_price_repo.mid_price_signal = (current_bid_prices[0] + current_ask_prices[0]) * 0.5;
    raw_price_repo.vwap_bid_price_signal_0 = cum_bid_amounts[0] / cum_bid_sizes[0];
    raw_price_repo.vwap_bid_price_signal_1 = cum_bid_amounts[1] / cum_bid_sizes[1];
    raw_price_repo.vwap_bid_price_signal_2 = cum_bid_amounts[2] / cum_bid_sizes[2];
    raw_price_repo.vwap_bid_price_signal_3 = cum_bid_amounts[3] / cum_bid_sizes[3];
    raw_price_repo.vwap_bid_price_signal_4 = cum_bid_amounts[4] / cum_bid_sizes[4];

    raw_price_repo.vwap_ask_price_signal_0 = cum_ask_amounts[0] / cum_ask_sizes[0];
    raw_price_repo.vwap_ask_price_signal_1 = cum_ask_amounts[1] / cum_ask_sizes[1];
    raw_price_repo.vwap_ask_price_signal_2 = cum_ask_amounts[2] / cum_ask_sizes[2];
    raw_price_repo.vwap_ask_price_signal_3 = cum_ask_amounts[3] / cum_ask_sizes[3];
    raw_price_repo.vwap_ask_price_signal_4 = cum_ask_amounts[4] / cum_ask_sizes[4];

    auto cum_mid_size_0 = (cum_bid_sizes[0] + cum_ask_sizes[0]) / 2.0;
    auto cum_mid_size_1 = (cum_bid_sizes[1] + cum_ask_sizes[1]) / 2.0;
    auto cum_mid_size_2 = (cum_bid_sizes[2] + cum_ask_sizes[2]) / 2.0;
    auto cum_mid_size_3 = (cum_bid_sizes[3] + cum_ask_sizes[3]) / 2.0;
    auto cum_mid_size_4 = (cum_bid_sizes[4] + cum_ask_sizes[4]) / 2.0;

    raw_price_repo.norm_vwap_bid_price_signal_0 = fill_price(current_bid_prices, current_bid_sizes, cum_mid_size_0);
    raw_price_repo.norm_vwap_bid_price_signal_1 = fill_price(current_bid_prices, current_bid_sizes, cum_mid_size_1);
    raw_price_repo.norm_vwap_bid_price_signal_2 = fill_price(current_bid_prices, current_bid_sizes, cum_mid_size_2);
    raw_price_repo.norm_vwap_bid_price_signal_3 = fill_price(current_bid_prices, current_bid_sizes, cum_mid_size_3);
    raw_price_repo.norm_vwap_bid_price_signal_4 = fill_price(current_bid_prices, current_bid_sizes, cum_mid_size_4);

    raw_price_repo.norm_vwap_ask_price_signal_0 = fill_price(current_ask_prices, current_ask_sizes, cum_mid_size_0);
    raw_price_repo.norm_vwap_ask_price_signal_1 = fill_price(current_ask_prices, current_ask_sizes, cum_mid_size_1);
    raw_price_repo.norm_vwap_ask_price_signal_2 = fill_price(current_ask_prices, current_ask_sizes, cum_mid_size_2);
    raw_price_repo.norm_vwap_ask_price_signal_3 = fill_price(current_ask_prices, current_ask_sizes, cum_mid_size_3);
    raw_price_repo.norm_vwap_ask_price_signal_4 = fill_price(current_ask_prices, current_ask_sizes, cum_mid_size_4);

    // compute micro signals
    raw_price_repo.micro_price_signal_0 = micro_price(raw_price_repo.norm_vwap_bid_price_signal_0,
                                                      raw_price_repo.norm_vwap_ask_price_signal_0,
                                                      cum_bid_sizes[0], cum_ask_sizes[0]);
    raw_price_repo.micro_price_signal_1 = micro_price(raw_price_repo.norm_vwap_bid_price_signal_1,
                                                      raw_price_repo.norm_vwap_ask_price_signal_1,
                                                      cum_bid_sizes[1], cum_ask_sizes[1]);
    raw_price_repo.micro_price_signal_2 = micro_price(raw_price_repo.norm_vwap_bid_price_signal_2,
                                                      raw_price_repo.norm_vwap_ask_price_signal_2,
                                                      cum_bid_sizes[2], cum_ask_sizes[2]);
    raw_price_repo.micro_price_signal_3 = micro_price(raw_price_repo.norm_vwap_bid_price_signal_3,
                                                      raw_price_repo.norm_vwap_ask_price_signal_3,
                                                      cum_bid_sizes[3], cum_ask_sizes[3]);
    raw_price_repo.micro_price_signal_4 = micro_price(raw_price_repo.norm_vwap_bid_price_signal_4,
                                                      raw_price_repo.norm_vwap_ask_price_signal_4,
                                                      cum_bid_sizes[4], cum_ask_sizes[4]);


    raw_price_repo.bid_fill_price_signal_0 = fill_price(current_bid_prices, current_bid_sizes, cum_ask_sizes[0]);
    raw_price_repo.bid_fill_price_signal_1 = fill_price(current_bid_prices, current_bid_sizes, cum_ask_sizes[1]);
    raw_price_repo.bid_fill_price_signal_2 = fill_price(current_bid_prices, current_bid_sizes, cum_ask_sizes[2]);
    raw_price_repo.bid_fill_price_signal_3 = fill_price(current_bid_prices, current_bid_sizes, cum_ask_sizes[3]);
    raw_price_repo.bid_fill_price_signal_4 = fill_price(current_bid_prices, current_bid_sizes, cum_ask_sizes[4]);

    raw_price_repo.ask_fill_price_signal_0 = fill_price(current_ask_prices, current_ask_sizes, cum_bid_sizes[0]);
    raw_price_repo.ask_fill_price_signal_1 = fill_price(current_ask_prices, current_ask_sizes, cum_bid_sizes[1]);
    raw_price_repo.ask_fill_price_signal_2 = fill_price(current_ask_prices, current_ask_sizes, cum_bid_sizes[2]);
    raw_price_repo.ask_fill_price_signal_3 = fill_price(current_ask_prices, current_ask_sizes, cum_bid_sizes[3]);
    raw_price_repo.ask_fill_price_signal_4 = fill_price(current_ask_prices, current_ask_sizes, cum_bid_sizes[4]);

    raw_price_diff_signals->mid_price_signal = raw_price_repo.mid_price_signal - previous_price_signal.mid_price_signal;
    raw_price_diff_signals->vwap_bid_price_signal_0 = raw_price_repo.vwap_bid_price_signal_0 - previous_price_signal.vwap_bid_price_signal_0;
    raw_price_diff_signals->vwap_bid_price_signal_1 = raw_price_repo.vwap_bid_price_signal_1 - previous_price_signal.vwap_bid_price_signal_1;
    raw_price_diff_signals->vwap_bid_price_signal_2 = raw_price_repo.vwap_bid_price_signal_2 - previous_price_signal.vwap_bid_price_signal_2;
    raw_price_diff_signals->vwap_bid_price_signal_3 = raw_price_repo.vwap_bid_price_signal_3 - previous_price_signal.vwap_bid_price_signal_3;
    raw_price_diff_signals->vwap_bid_price_signal_4 = raw_price_repo.vwap_bid_price_signal_4 - previous_price_signal.vwap_bid_price_signal_4;

    raw_price_diff_signals->vwap_ask_price_signal_0 = raw_price_repo.vwap_ask_price_signal_0 - previous_price_signal.vwap_ask_price_signal_0;
    raw_price_diff_signals->vwap_ask_price_signal_1 = raw_price_repo.vwap_ask_price_signal_1 - previous_price_signal.vwap_ask_price_signal_1;
    raw_price_diff_signals->vwap_ask_price_signal_2 = raw_price_repo.vwap_ask_price_signal_2 - previous_price_signal.vwap_ask_price_signal_2;
    raw_price_diff_signals->vwap_ask_price_signal_3 = raw_price_repo.vwap_ask_price_signal_3 - previous_price_signal.vwap_ask_price_signal_3;
    raw_price_diff_signals->vwap_ask_price_signal_4 = raw_price_repo.vwap_ask_price_signal_4 - previous_price_signal.vwap_ask_price_signal_4;

    raw_price_diff_signals->norm_vwap_bid_price_signal_0 = raw_price_repo.norm_vwap_bid_price_signal_0 - previous_price_signal.norm_vwap_bid_price_signal_0;
    raw_price_diff_signals->norm_vwap_bid_price_signal_1 = raw_price_repo.norm_vwap_bid_price_signal_1 - previous_price_signal.norm_vwap_bid_price_signal_1;
    raw_price_diff_signals->norm_vwap_bid_price_signal_2 = raw_price_repo.norm_vwap_bid_price_signal_2 - previous_price_signal.norm_vwap_bid_price_signal_2;
    raw_price_diff_signals->norm_vwap_bid_price_signal_3 = raw_price_repo.norm_vwap_bid_price_signal_3 - previous_price_signal.norm_vwap_bid_price_signal_3;
    raw_price_diff_signals->norm_vwap_bid_price_signal_4 = raw_price_repo.norm_vwap_bid_price_signal_4 - previous_price_signal.norm_vwap_bid_price_signal_4;

    raw_price_diff_signals->norm_vwap_ask_price_signal_0 = raw_price_repo.norm_vwap_ask_price_signal_0 - previous_price_signal.norm_vwap_ask_price_signal_0;
    raw_price_diff_signals->norm_vwap_ask_price_signal_1 = raw_price_repo.norm_vwap_ask_price_signal_1 - previous_price_signal.norm_vwap_ask_price_signal_1;
    raw_price_diff_signals->norm_vwap_ask_price_signal_2 = raw_price_repo.norm_vwap_ask_price_signal_2 - previous_price_signal.norm_vwap_ask_price_signal_2;
    raw_price_diff_signals->norm_vwap_ask_price_signal_3 = raw_price_repo.norm_vwap_ask_price_signal_3 - previous_price_signal.norm_vwap_ask_price_signal_3;
    raw_price_diff_signals->norm_vwap_ask_price_signal_4 = raw_price_repo.norm_vwap_ask_price_signal_4 - previous_price_signal.norm_vwap_ask_price_signal_4;

    raw_price_diff_signals->micro_price_signal_0 = raw_price_repo.micro_price_signal_0 - previous_price_signal.micro_price_signal_0;
    raw_price_diff_signals->micro_price_signal_1 = raw_price_repo.micro_price_signal_1 - previous_price_signal.micro_price_signal_1;
    raw_price_diff_signals->micro_price_signal_2 = raw_price_repo.micro_price_signal_2 - previous_price_signal.micro_price_signal_2;
    raw_price_diff_signals->micro_price_signal_3 = raw_price_repo.micro_price_signal_3 - previous_price_signal.micro_price_signal_3;
    raw_price_diff_signals->micro_price_signal_4 = raw_price_repo.micro_price_signal_4 - previous_price_signal.micro_price_signal_4;

    raw_price_diff_signals->bid_fill_price_signal_0 = raw_price_repo.bid_fill_price_signal_0 - previous_price_signal.bid_fill_price_signal_0;
    raw_price_diff_signals->bid_fill_price_signal_1 = raw_price_repo.bid_fill_price_signal_1 - previous_price_signal.bid_fill_price_signal_1;
    raw_price_diff_signals->bid_fill_price_signal_2 = raw_price_repo.bid_fill_price_signal_2 - previous_price_signal.bid_fill_price_signal_2;
    raw_price_diff_signals->bid_fill_price_signal_3 = raw_price_repo.bid_fill_price_signal_3 - previous_price_signal.bid_fill_price_signal_3;
    raw_price_diff_signals->bid_fill_price_signal_4 = raw_price_repo.bid_fill_price_signal_4 - previous_price_signal.bid_fill_price_signal_4;

    raw_price_diff_signals->ask_fill_price_signal_0 = raw_price_repo.ask_fill_price_signal_0 - previous_price_signal.ask_fill_price_signal_0;
    raw_price_diff_signals->ask_fill_price_signal_1 = raw_price_repo.ask_fill_price_signal_1 - previous_price_signal.ask_fill_price_signal_1;
    raw_price_diff_signals->ask_fill_price_signal_2 = raw_price_repo.ask_fill_price_signal_2 - previous_price_signal.ask_fill_price_signal_2;
    raw_price_diff_signals->ask_fill_price_signal_3 = raw_price_repo.ask_fill_price_signal_3 - previous_price_signal.ask_fill_price_signal_3;
    raw_price_diff_signals->ask_fill_price_signal_4 = raw_price_repo.ask_fill_price_signal_4 - previous_price_signal.ask_fill_price_signal_4;

    previous_price_signal = raw_price_repo;
}

void compute_ewma_lagged(double lag, price_signal_repository& ewma_price_signal, const price_signal_repository& repo) {
    double ewma = 2.0 / (1 + lag);
    double ewma_rolling = 1.0 - ewma;

    ewma_price_signal.mid_price_signal = ewma_rolling * ewma_price_signal.mid_price_signal + ewma * repo.mid_price_signal;

    ewma_price_signal.vwap_bid_price_signal_0 = ewma_rolling * ewma_price_signal.vwap_bid_price_signal_0 + ewma * repo.vwap_bid_price_signal_0;
    ewma_price_signal.vwap_bid_price_signal_1 = ewma_rolling * ewma_price_signal.vwap_bid_price_signal_1 + ewma * repo.vwap_bid_price_signal_1;
    ewma_price_signal.vwap_bid_price_signal_2 = ewma_rolling * ewma_price_signal.vwap_bid_price_signal_2 + ewma * repo.vwap_bid_price_signal_2;
    ewma_price_signal.vwap_bid_price_signal_3 = ewma_rolling * ewma_price_signal.vwap_bid_price_signal_3 + ewma * repo.vwap_bid_price_signal_3;
    ewma_price_signal.vwap_bid_price_signal_4 = ewma_rolling * ewma_price_signal.vwap_bid_price_signal_4 + ewma * repo.vwap_bid_price_signal_4;

    ewma_price_signal.vwap_ask_price_signal_0 = ewma_rolling * ewma_price_signal.vwap_ask_price_signal_0 + ewma * repo.vwap_ask_price_signal_0;
    ewma_price_signal.vwap_ask_price_signal_1 = ewma_rolling * ewma_price_signal.vwap_ask_price_signal_1 + ewma * repo.vwap_ask_price_signal_1;
    ewma_price_signal.vwap_ask_price_signal_2 = ewma_rolling * ewma_price_signal.vwap_ask_price_signal_2 + ewma * repo.vwap_ask_price_signal_2;
    ewma_price_signal.vwap_ask_price_signal_3 = ewma_rolling * ewma_price_signal.vwap_ask_price_signal_3 + ewma * repo.vwap_ask_price_signal_3;
    ewma_price_signal.vwap_ask_price_signal_4 = ewma_rolling * ewma_price_signal.vwap_ask_price_signal_4 + ewma * repo.vwap_ask_price_signal_4;

    ewma_price_signal.micro_price_signal_0 = ewma_rolling * ewma_price_signal.micro_price_signal_0 + ewma * repo.micro_price_signal_0;
    ewma_price_signal.micro_price_signal_0 = ewma_rolling * ewma_price_signal.micro_price_signal_1 + ewma * repo.micro_price_signal_1;
    ewma_price_signal.micro_price_signal_0 = ewma_rolling * ewma_price_signal.micro_price_signal_2 + ewma * repo.micro_price_signal_2;
    ewma_price_signal.micro_price_signal_0 = ewma_rolling * ewma_price_signal.micro_price_signal_3 + ewma * repo.micro_price_signal_3;
    ewma_price_signal.micro_price_signal_0 = ewma_rolling * ewma_price_signal.micro_price_signal_4 + ewma * repo.micro_price_signal_4;

    ewma_price_signal.norm_vwap_bid_price_signal_0 = ewma_rolling * ewma_price_signal.norm_vwap_bid_price_signal_0 + ewma * repo.norm_vwap_bid_price_signal_0;
    ewma_price_signal.norm_vwap_bid_price_signal_1 = ewma_rolling * ewma_price_signal.norm_vwap_bid_price_signal_1 + ewma * repo.norm_vwap_bid_price_signal_1;
    ewma_price_signal.norm_vwap_bid_price_signal_2 = ewma_rolling * ewma_price_signal.norm_vwap_bid_price_signal_2 + ewma * repo.norm_vwap_bid_price_signal_2;
    ewma_price_signal.norm_vwap_bid_price_signal_3 = ewma_rolling * ewma_price_signal.norm_vwap_bid_price_signal_3 + ewma * repo.norm_vwap_bid_price_signal_3;
    ewma_price_signal.norm_vwap_bid_price_signal_4 = ewma_rolling * ewma_price_signal.norm_vwap_bid_price_signal_4 + ewma * repo.norm_vwap_bid_price_signal_4;

    ewma_price_signal.norm_vwap_ask_price_signal_0 = ewma_rolling * ewma_price_signal.norm_vwap_ask_price_signal_0 + ewma * repo.norm_vwap_ask_price_signal_0;
    ewma_price_signal.norm_vwap_ask_price_signal_1 = ewma_rolling * ewma_price_signal.norm_vwap_ask_price_signal_1 + ewma * repo.norm_vwap_ask_price_signal_1;
    ewma_price_signal.norm_vwap_ask_price_signal_2 = ewma_rolling * ewma_price_signal.norm_vwap_ask_price_signal_2 + ewma * repo.norm_vwap_ask_price_signal_2;
    ewma_price_signal.norm_vwap_ask_price_signal_3 = ewma_rolling * ewma_price_signal.norm_vwap_ask_price_signal_3 + ewma * repo.norm_vwap_ask_price_signal_3;
    ewma_price_signal.norm_vwap_ask_price_signal_4 = ewma_rolling * ewma_price_signal.norm_vwap_ask_price_signal_4 + ewma * repo.norm_vwap_ask_price_signal_4;

    ewma_price_signal.bid_fill_price_signal_0 = ewma_rolling * ewma_price_signal.bid_fill_price_signal_0 + ewma * repo.bid_fill_price_signal_0;
    ewma_price_signal.bid_fill_price_signal_1 = ewma_rolling * ewma_price_signal.bid_fill_price_signal_1 + ewma * repo.bid_fill_price_signal_1;
    ewma_price_signal.bid_fill_price_signal_2 = ewma_rolling * ewma_price_signal.bid_fill_price_signal_2 + ewma * repo.bid_fill_price_signal_2;
    ewma_price_signal.bid_fill_price_signal_3 = ewma_rolling * ewma_price_signal.bid_fill_price_signal_3 + ewma * repo.bid_fill_price_signal_3;
    ewma_price_signal.bid_fill_price_signal_4 = ewma_rolling * ewma_price_signal.bid_fill_price_signal_4 + ewma * repo.bid_fill_price_signal_4;

    ewma_price_signal.ask_fill_price_signal_0 = ewma_rolling * ewma_price_signal.ask_fill_price_signal_0 + ewma * repo.ask_fill_price_signal_0;
    ewma_price_signal.ask_fill_price_signal_1 = ewma_rolling * ewma_price_signal.ask_fill_price_signal_1 + ewma * repo.ask_fill_price_signal_1;
    ewma_price_signal.ask_fill_price_signal_2 = ewma_rolling * ewma_price_signal.ask_fill_price_signal_2 + ewma * repo.ask_fill_price_signal_2;
    ewma_price_signal.ask_fill_price_signal_3 = ewma_rolling * ewma_price_signal.ask_fill_price_signal_3 + ewma * repo.ask_fill_price_signal_3;
    ewma_price_signal.ask_fill_price_signal_4 = ewma_rolling * ewma_price_signal.ask_fill_price_signal_4 + ewma * repo.ask_fill_price_signal_4;
}

void norm_ewma_signals(price_signal_repository& output, price_signal_repository& value, const price_signal_repository& base) {
    output.mid_price_signal = base.mid_price_signal - value.mid_price_signal;
    if (std::abs(value.mid_price_signal) > 0) output.mid_price_signal /= std::abs(value.mid_price_signal);

    output.vwap_bid_price_signal_0 = base.vwap_bid_price_signal_0 - value.vwap_bid_price_signal_0;
    if (std::abs(value.vwap_bid_price_signal_0) > 0) output.vwap_bid_price_signal_0 /= std::abs(value.vwap_bid_price_signal_0);
    output.vwap_bid_price_signal_1 = base.vwap_bid_price_signal_1 - value.vwap_bid_price_signal_1;
    if (std::abs(value.vwap_bid_price_signal_1) > 0) output.vwap_bid_price_signal_1 /= std::abs(value.vwap_bid_price_signal_1);
    output.vwap_bid_price_signal_2 = base.vwap_bid_price_signal_2 - value.vwap_bid_price_signal_2;
    if (std::abs(value.vwap_bid_price_signal_2) > 0) output.vwap_bid_price_signal_2 /= std::abs(value.vwap_bid_price_signal_2);
    output.vwap_bid_price_signal_3 = base.vwap_bid_price_signal_3 - value.vwap_bid_price_signal_3;
    if (std::abs(value.vwap_bid_price_signal_3) > 0) output.vwap_bid_price_signal_3 /= std::abs(value.vwap_bid_price_signal_3);
    output.vwap_bid_price_signal_4 = base.vwap_bid_price_signal_4 - value.vwap_bid_price_signal_4;
    if (std::abs(value.vwap_bid_price_signal_4) > 0) output.vwap_bid_price_signal_4 /= std::abs(value.vwap_bid_price_signal_4);

    output.vwap_ask_price_signal_0 = base.vwap_ask_price_signal_0 - value.vwap_ask_price_signal_0;
    if (std::abs(value.vwap_ask_price_signal_0) > 0) output.vwap_ask_price_signal_0 /= std::abs(value.vwap_ask_price_signal_0);
    output.vwap_ask_price_signal_1 = base.vwap_ask_price_signal_1 - value.vwap_ask_price_signal_1;
    if (std::abs(value.vwap_ask_price_signal_1) > 0) output.vwap_ask_price_signal_1 /= std::abs(value.vwap_ask_price_signal_1);
    output.vwap_ask_price_signal_2 = base.vwap_ask_price_signal_2 - value.vwap_ask_price_signal_2;
    if (std::abs(value.vwap_ask_price_signal_2) > 0) output.vwap_ask_price_signal_2 /= std::abs(value.vwap_ask_price_signal_2);
    output.vwap_ask_price_signal_3 = base.vwap_ask_price_signal_3 - value.vwap_ask_price_signal_3;
    if (std::abs(value.vwap_ask_price_signal_3) > 0) output.vwap_ask_price_signal_3 /= std::abs(value.vwap_ask_price_signal_3);
    output.vwap_ask_price_signal_4 = base.vwap_ask_price_signal_4 - value.vwap_ask_price_signal_4;
    if (std::abs(value.vwap_ask_price_signal_4) > 0) output.vwap_ask_price_signal_4 /= std::abs(value.vwap_ask_price_signal_4);

    output.micro_price_signal_0 = base.micro_price_signal_0 - value.micro_price_signal_0;
    if (std::abs(value.micro_price_signal_0) > 0) output.micro_price_signal_0 /= std::abs(value.micro_price_signal_0);
    output.micro_price_signal_1 = base.micro_price_signal_1 - value.micro_price_signal_1;
    if (std::abs(value.micro_price_signal_1) > 0) output.micro_price_signal_1 /= std::abs(value.micro_price_signal_1);
    output.micro_price_signal_2 = base.micro_price_signal_2 - value.micro_price_signal_2;
    if (std::abs(value.micro_price_signal_2) > 0) output.micro_price_signal_2 /= std::abs(value.micro_price_signal_2);
    output.micro_price_signal_3 = base.micro_price_signal_3 - value.micro_price_signal_3;
    if (std::abs(value.micro_price_signal_3) > 0) output.micro_price_signal_3 /= std::abs(value.micro_price_signal_3);
    output.micro_price_signal_4 = base.micro_price_signal_4 - value.micro_price_signal_4;
    if (std::abs(value.micro_price_signal_4) > 0) output.micro_price_signal_4 /= std::abs(value.micro_price_signal_4);

    output.norm_vwap_bid_price_signal_0 = base.norm_vwap_bid_price_signal_0 - value.norm_vwap_bid_price_signal_0;
    if (std::abs(value.norm_vwap_bid_price_signal_0) > 0) output.norm_vwap_bid_price_signal_0 /= std::abs(value.norm_vwap_bid_price_signal_0);
    output.norm_vwap_bid_price_signal_1 = base.norm_vwap_bid_price_signal_1 - value.norm_vwap_bid_price_signal_1;
    if (std::abs(value.norm_vwap_bid_price_signal_1) > 0) output.norm_vwap_bid_price_signal_1 /= std::abs(value.norm_vwap_bid_price_signal_1);
    output.norm_vwap_bid_price_signal_2 = base.norm_vwap_bid_price_signal_2 - value.norm_vwap_bid_price_signal_2;
    if (std::abs(value.norm_vwap_bid_price_signal_2) > 0) output.norm_vwap_bid_price_signal_2 /= std::abs(value.norm_vwap_bid_price_signal_2);
    output.norm_vwap_bid_price_signal_3 = base.norm_vwap_bid_price_signal_3 - value.norm_vwap_bid_price_signal_3;
    if (std::abs(value.norm_vwap_bid_price_signal_3) > 0) output.norm_vwap_bid_price_signal_3 /= std::abs(value.norm_vwap_bid_price_signal_3);
    output.norm_vwap_bid_price_signal_4 = base.norm_vwap_bid_price_signal_4 - value.norm_vwap_bid_price_signal_4;
    if (std::abs(value.norm_vwap_bid_price_signal_4) > 0) output.norm_vwap_bid_price_signal_4 /= std::abs(value.norm_vwap_bid_price_signal_4);

    output.norm_vwap_ask_price_signal_0 = base.norm_vwap_ask_price_signal_0 - value.norm_vwap_ask_price_signal_0;
    if (std::abs(value.norm_vwap_ask_price_signal_0) > 0) output.norm_vwap_ask_price_signal_0 /= std::abs(value.norm_vwap_ask_price_signal_0);
    output.norm_vwap_ask_price_signal_1 = base.norm_vwap_ask_price_signal_1 - value.norm_vwap_ask_price_signal_1;
    if (std::abs(value.norm_vwap_ask_price_signal_1) > 0) output.norm_vwap_ask_price_signal_1 /= std::abs(value.norm_vwap_ask_price_signal_1);
    output.norm_vwap_ask_price_signal_2 = base.norm_vwap_ask_price_signal_2 - value.norm_vwap_ask_price_signal_2;
    if (std::abs(value.norm_vwap_ask_price_signal_2) > 0) output.norm_vwap_ask_price_signal_2 /= std::abs(value.norm_vwap_ask_price_signal_2);
    output.norm_vwap_ask_price_signal_3 = base.norm_vwap_ask_price_signal_3 - value.norm_vwap_ask_price_signal_3;
    if (std::abs(value.norm_vwap_ask_price_signal_3) > 0) output.norm_vwap_ask_price_signal_3 /= std::abs(value.norm_vwap_ask_price_signal_3);
    output.norm_vwap_ask_price_signal_4 = base.norm_vwap_ask_price_signal_4 - value.norm_vwap_ask_price_signal_4;
    if (std::abs(value.norm_vwap_ask_price_signal_4) > 0) output.norm_vwap_ask_price_signal_4 /= std::abs(value.norm_vwap_ask_price_signal_4);

    output.bid_fill_price_signal_0 = base.bid_fill_price_signal_0 - value.bid_fill_price_signal_0;
    if (std::abs(value.bid_fill_price_signal_0) > 0) output.bid_fill_price_signal_0 /= std::abs(value.bid_fill_price_signal_0);
    output.bid_fill_price_signal_1 = base.bid_fill_price_signal_1 - value.bid_fill_price_signal_1;
    if (std::abs(value.bid_fill_price_signal_1) > 0) output.bid_fill_price_signal_1 /= std::abs(value.bid_fill_price_signal_1);
    output.bid_fill_price_signal_2 = base.bid_fill_price_signal_2 - value.bid_fill_price_signal_2;
    if (std::abs(value.bid_fill_price_signal_2) > 0) output.bid_fill_price_signal_2 /= std::abs(value.bid_fill_price_signal_2);
    output.bid_fill_price_signal_3 = base.bid_fill_price_signal_3 - value.bid_fill_price_signal_3;
    if (std::abs(value.bid_fill_price_signal_3) > 0) output.bid_fill_price_signal_3 /= std::abs(value.bid_fill_price_signal_3);
    output.bid_fill_price_signal_4 = base.bid_fill_price_signal_4 - value.bid_fill_price_signal_4;
    if (std::abs(value.bid_fill_price_signal_4) > 0) output.bid_fill_price_signal_4 /= std::abs(value.bid_fill_price_signal_4);

    output.ask_fill_price_signal_0 = base.ask_fill_price_signal_0 - value.ask_fill_price_signal_0;
    if (std::abs(value.ask_fill_price_signal_0) > 0) output.ask_fill_price_signal_0 /= std::abs(value.ask_fill_price_signal_0);
    output.ask_fill_price_signal_1 = base.ask_fill_price_signal_1 - value.ask_fill_price_signal_1;
    if (std::abs(value.ask_fill_price_signal_1) > 0) output.ask_fill_price_signal_1 /= std::abs(value.ask_fill_price_signal_1);
    output.ask_fill_price_signal_2 = base.ask_fill_price_signal_2 - value.ask_fill_price_signal_2;
    if (std::abs(value.ask_fill_price_signal_2) > 0) output.ask_fill_price_signal_2 /= std::abs(value.ask_fill_price_signal_2);
    output.ask_fill_price_signal_3 = base.ask_fill_price_signal_3 - value.ask_fill_price_signal_3;
    if (std::abs(value.ask_fill_price_signal_3) > 0) output.ask_fill_price_signal_3 /= std::abs(value.ask_fill_price_signal_3);
    output.ask_fill_price_signal_4 = base.ask_fill_price_signal_4 - value.ask_fill_price_signal_4;
    if (std::abs(value.ask_fill_price_signal_4) > 0) output.ask_fill_price_signal_4 /= std::abs(value.ask_fill_price_signal_4);
}

void MarketSignalBuilder::compute_ewma_signals(const price_signal_repository& repo) {
    compute_ewma_lagged(30, ewma_price_signal_30, repo);
    compute_ewma_lagged(60, ewma_price_signal_60, repo);
    compute_ewma_lagged(120, ewma_price_signal_120, repo);
    compute_ewma_lagged(300, ewma_price_signal_300, repo);
    norm_ewma_signals(*ewma_diff_signals_30, ewma_price_signal_30, repo);
    norm_ewma_signals(*ewma_diff_signals_60, ewma_price_signal_60, ewma_price_signal_30);
    norm_ewma_signals(*ewma_diff_signals_120, ewma_price_signal_120, ewma_price_signal_60);
    norm_ewma_signals(*ewma_diff_signals_300, ewma_price_signal_300, ewma_price_signal_120);
}
