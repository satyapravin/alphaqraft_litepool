#include "position_signal_builder.h"

#include "market_signal_builder.h"
#include "rl_macros.h"

using namespace RLTrader;

PositionSignalBuilder::PositionSignalBuilder()
                      :raw_previous_signals(std::make_unique<position_signal_repository>()),
                       raw_spread_signals(std::make_unique<position_signal_repository>()) {
}

std::vector<double> PositionSignalBuilder::add_info(const PositionInfo& info, const double& bid_price, const double& ask_price) {
    compute_signals(info, bid_price, ask_price);

    std::vector<double> retval;
    insert_signals(retval, *raw_spread_signals);
    insert_signals(retval, *raw_previous_signals);
    return retval;
}

void PositionSignalBuilder::compute_signals(const PositionInfo& info, const double& bid_price, const double& ask_price) {
    position_signal_repository repo;
    position_signal_repository& previous_repo = *raw_previous_signals;
    repo.net_position = info.netPosition;
    repo.inventory_pnl = info.inventoryPnL;
    repo.realized_pnl = info.tradingPnL;

    if (max_inventory_pnl < repo.inventory_pnl) max_inventory_pnl = repo.inventory_pnl;
    if (max_trading_pnl < repo.realized_pnl) max_trading_pnl = repo.realized_pnl;
    repo.inventory_pnl_drawdown = std::min(repo.inventory_pnl - max_inventory_pnl, 0.0);
    repo.realized_pnl_drawdown = std::min(repo.realized_pnl - max_trading_pnl, 0.0);
    repo.total_pnl = repo.inventory_pnl + repo.realized_pnl;
    repo.total_drawdown = repo.inventory_pnl_drawdown + repo.realized_pnl_drawdown;

    if (info.netPosition > 0) repo.relative_price = ask_price - info.averagePrice;
    else if (info.netPosition < 0) repo.relative_price = info.averagePrice - bid_price;
    else repo.relative_price = 0.0; 

    repo.net_position = repo.net_position / info.balance;
    repo.inventory_pnl = repo.inventory_pnl / info.balance;
    repo.realized_pnl = repo.realized_pnl / info.balance;
    repo.inventory_pnl_drawdown = repo.inventory_pnl_drawdown / info.balance;
    repo.realized_pnl_drawdown = repo.realized_pnl_drawdown / info.balance;
    repo.total_pnl = repo.total_pnl / info.balance;
    repo.total_drawdown = repo.total_drawdown / info.balance;
    repo.leverage = info.leverage;

    auto& spread_repo = *raw_spread_signals;
    spread_repo.inventory_pnl = repo.inventory_pnl - previous_repo.inventory_pnl;
    spread_repo.inventory_pnl_drawdown = repo.inventory_pnl_drawdown - previous_repo.inventory_pnl_drawdown;
    spread_repo.net_position = repo.net_position - previous_repo.net_position;
    spread_repo.realized_pnl_drawdown = repo.realized_pnl_drawdown - previous_repo.realized_pnl_drawdown;
    spread_repo.realized_pnl = repo.realized_pnl - previous_repo.realized_pnl;
    spread_repo.total_pnl = repo.total_pnl - previous_repo.total_pnl;
    spread_repo.total_drawdown = repo.total_drawdown - previous_repo.total_drawdown;
    spread_repo.relative_price = repo.relative_price - previous_repo.relative_price;
    spread_repo.leverage = repo.leverage - previous_repo.leverage;

    previous_repo.leverage = repo.leverage;
    previous_repo.inventory_pnl = repo.inventory_pnl;
    previous_repo.inventory_pnl_drawdown = repo.inventory_pnl_drawdown;
    previous_repo.net_position = repo.net_position;
    previous_repo.realized_pnl_drawdown = repo.realized_pnl_drawdown;
    previous_repo.realized_pnl = repo.realized_pnl;
    previous_repo.total_pnl = repo.total_pnl;
    previous_repo.total_drawdown = repo.total_drawdown;
    previous_repo.relative_price = repo.relative_price;
}
