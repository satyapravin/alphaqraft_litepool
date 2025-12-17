// Copyright 2021 Garena Online Private Limited
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

#ifndef LITEPOOL_RLTRADER_RLTRADER_LITEPOOL_H_
#define LITEPOOL_RLTRADER_RLTRADER_LITEPOOL_H_

#include <memory>
#include <vector>
#include <algorithm>
#include <tuple>
#include "litepool/core/async_litepool.h"
#include "litepool/core/env.h"
#include "env_adaptor.h"

#include "base_instrument.h"
#include "inverse_instrument.h"
#include "normal_instrument.h"

#include <filesystem>

#include "crypto_exchange.h"
#include "sim_exchange.h"

namespace fs = std::filesystem;
namespace rltrader {

class RlTraderEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("is_prod"_.Bind<bool>(false),
                    "api_key"_.Bind(std::string("")),
                    "api_secret"_.Bind(std::string("")),
                    "is_inverse_instr"_.Bind<bool>(true),
                    "symbol"_.Bind((std::string(""))),
                    "hedge_symbol"_.Bind((std::string(""))),
                    "tick_size"_.Bind<double>(0.5),
                    "min_amount"_.Bind<double>(10.0),
                    "maker_fee"_.Bind<double>(-0.0001),
                    "taker_fee"_.Bind<double>(0.0005),
                    "foldername"_.Bind(std::string("./train_files/")),
                    "balance"_.Bind(1.0),
                    "start"_.Bind<int>(0),
                    "max"_.Bind<int>(72000),
                    "ticks_per_step"_.Bind<int>(5),  // Advance 5 ticks per RL step
                    "base_spread_bps"_.Bind<double>(1.0),  // Base spread in basis points
                    "min_size_pct"_.Bind<double>(0.5),      // Minimum order size as % of balance
                    "max_size_pct"_.Bind<double>(2.0));    // Maximum order size as % of balance
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<double>({16})),  // 13 market signals + 3 AMM flow signals
                    "info:mid_price"_.Bind(Spec<double>({-1})),
                    "info:balance"_.Bind(Spec<double>({-1})),
                    "info:unrealized_pnl"_.Bind(Spec<double>({-1})),
                    "info:realized_pnl"_.Bind(Spec<double>({-1})),
                    "info:leverage"_.Bind(Spec<double>({-1})),
                    "info:trade_count"_.Bind(Spec<double>({-1})),
                    "info:buy_trades"_.Bind(Spec<double>({-1})),
                    "info:sell_trades"_.Bind(Spec<double>({-1})),
                    "info:buy_amount"_.Bind(Spec<double>({-1})),
                    "info:sell_amount"_.Bind(Spec<double>({-1})),
                    "info:drawdown"_.Bind(Spec<double>({-1})),
                    "info:fees"_.Bind((Spec<double>({-1}))),
                    "info:mid_diff"_.Bind((Spec<double>({-1}))),
                    "info:done"_.Bind((Spec<bool>({-1}))),
                    "info:net_position_usd"_.Bind(Spec<double>({-1})),
                    "info:net_amount_btc"_.Bind(Spec<double>({-1})),
                    "info:last_bid_price"_.Bind(Spec<double>({-1})),
                    "info:last_ask_price"_.Bind(Spec<double>({-1})),
                    "info:last_mid_price"_.Bind(Spec<double>({-1})),
                    // Terminal info from completed episode (available after auto-reset)
                    "info:final_realized_pnl"_.Bind(Spec<double>({-1})),
                    "info:final_unrealized_pnl"_.Bind(Spec<double>({-1})),
                    "info:final_trade_count"_.Bind(Spec<double>({-1})),
                    "info:final_fees"_.Bind(Spec<double>({-1})),
                    "info:final_net_amount_btc"_.Bind(Spec<double>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // 5-action space: spread, size, skew, target_inventory, should_requote
    return MakeDict("action"_.Bind(Spec<float>({5}, {{ -1., -1., -1., -1., -1. },
                                                     {  1.,  1.,  1.,  1.,  1. }})));
  }
};


using RlTraderEnvSpec = EnvSpec<RlTraderEnvFns>;

class RlTraderEnv : public Env<RlTraderEnvSpec> {
 protected:
  int state_{0};
  bool isDone = true;
  bool is_prod = false;
  bool is_inverse_instr = false;
  std::string api_key;
  std::string api_secret;
  std::string symbol;
  std::string hedge_symbol;
  double tick_size;
  double min_amount;
  double maker_fee;
  double taker_fee;
  std::string foldername;
  double balance = 0;
  int start_read = 0;
  int max_read = 0;
  int ticks_per_step = 5;
  double base_spread_bps = 0.0;
  double min_size_pct = 0.5;
  double max_size_pct = 2.0;
  long long steps = 0;
  
  // Reward tracking
  double prev_realized_pnl = 0.0;
  double prev_unrealized_from_anchor = 0.0;
  double prev_fees = 0.0;  // Track fees for fee rebate reward
  double price_anchor_ma = 0.0;  // Rolling MA of price (slow-moving "fair value")
  bool price_anchor_initialized = false;
  double initial_balance_ = 0.0; // Store initial balance for consistent reward scaling
  
  // Reward hyperparameters
  static constexpr double PRICE_MA_ALPHA = 0.002;     // Slow MA: ~500 step half-life for unrealized PnL anchor
  
  // Terminal info cache (stores metrics before reset for episode logging)
  std::unordered_map<std::string, double> terminal_info_;
  bool has_terminal_info_ = false;
  
  // Track fills from previous step to force requote if orders were filled
  bool had_fills_prev_step_ = false;
  
  // Track if RL chose to requote (not forced by auto-requote logic) for penalty
  // Removed: rl_chose_requote_ - was used for requote penalty which had no effect

  std::unique_ptr<RLTrader::BaseInstrument> instr_ptr;
  std::unique_ptr<RLTrader::BaseExchange> exchange_ptr;
  std::unique_ptr<RLTrader::Strategy> strategy_ptr;
  std::unique_ptr<RLTrader::EnvAdaptor> adaptor_ptr;
 public:
  RlTraderEnv(const Spec& spec, int env_id) : Env<RlTraderEnvSpec>(spec, env_id),
                                              is_prod(spec.config["is_prod"_]),
                                              is_inverse_instr(spec.config["is_inverse_instr"_]),
                                              api_key(spec.config["api_key"_]),
                                              api_secret(spec.config["api_secret"_]),
                                              symbol(spec.config["symbol"_]),
                                              hedge_symbol(spec.config["hedge_symbol"_]),
                                              tick_size(spec.config["tick_size"_]),
                                              min_amount(spec.config["min_amount"_]),
                                              maker_fee(spec.config["maker_fee"_]),
                                              taker_fee(spec.config["taker_fee"_]),
                                              foldername(spec.config["foldername"_]),
                                              balance(spec.config["balance"_]),
                                              start_read(spec.config["start"_]),
                                              max_read(spec.config["max"_]),
                                              ticks_per_step(spec.config["ticks_per_step"_]),
                                              base_spread_bps(spec.config["base_spread_bps"_]),
                                              min_size_pct(spec.config["min_size_pct"_]),
                                              max_size_pct(spec.config["max_size_pct"_])
  {

    RLTrader::BaseInstrument* instr_raw_ptr = nullptr;
    RLTrader::BaseExchange* exch_raw_ptr = nullptr;


    if (this->is_inverse_instr) {
      instr_raw_ptr = new RLTrader::InverseInstrument(symbol, tick_size, min_amount, maker_fee, taker_fee);
    } else {
      instr_raw_ptr = new RLTrader::NormalInstrument(symbol, tick_size, min_amount, maker_fee, taker_fee);
    }


    if (this->is_prod) {
      exch_raw_ptr = new RLTrader::CryptoExchange(symbol, hedge_symbol, api_key, api_secret);
    } else {
      int idx = env_id % 64;
      std::string filename = foldername + std::to_string(idx + 1) + ".csv";
      exch_raw_ptr = new RLTrader::SimExchange(filename, 250, start_read, max_read);
    }

    instr_ptr.reset(instr_raw_ptr);
    exchange_ptr.reset(exch_raw_ptr);
    RLTrader::StrategyConfig config;  // Use default config
    config.base_spread_bps = base_spread_bps;  // From Python config
    config.min_size_pct = min_size_pct;        // From Python config
    config.max_size_pct = max_size_pct;        // From Python config
    strategy_ptr = std::make_unique<RLTrader::Strategy>(*instr_ptr, *exchange_ptr, balance, 20, config);
    adaptor_ptr = std::make_unique<RLTrader::EnvAdaptor>(*strategy_ptr, *exchange_ptr, ticks_per_step);
  }

  void Reset() override {
    prev_realized_pnl = 0.0;
    prev_unrealized_from_anchor = 0.0;
    prev_fees = 0.0;
    price_anchor_ma = 0.0;
    price_anchor_initialized = false;
    initial_balance_ = balance;  // Store initial balance for consistent reward scaling
    steps = 0;
    had_fills_prev_step_ = false;  // Reset fill tracking
    adaptor_ptr->reset();
    isDone = false;
    // Note: Don't clear terminal_info_ here - it gets cleared after WriteState uses it
    WriteState();
  }

  void Step(const Action& action_dict) override { 
      RLTrader::RLAction action;
      // 5-action space: bid_spread, ask_spread, skew, target_inventory, should_requote
      action.bid_spread       = static_cast<double>(action_dict["action"_][0]);
      action.ask_spread       = static_cast<double>(action_dict["action"_][1]);
      action.skew             = static_cast<double>(action_dict["action"_][2]);
      action.target_inventory = static_cast<double>(action_dict["action"_][3]);
      action.should_requote   = static_cast<double>(action_dict["action"_][4]);
      
      // Update smoothed target inventory (EMA smoothing to prevent flickering)
      strategy_ptr->updateTargetInventory(action.target_inventory);
      
      // Force requote on first step (steps == 0) to place initial orders
      // After reset, there are no orders, so we must requote to place them
      if (steps == 0) {
          action.should_requote = 1.0;  // Force requote on first step
      }
      
      // Auto-requote if:
      // 1. No active orders in the market (orders were cancelled or never placed)
      // 2. Previous step had fills (orders were executed, need to replace them)
      bool has_active_orders = !exchange_ptr->getBidOrders().empty() || 
                               !exchange_ptr->getAskOrders().empty();
      bool forced_requote = (steps == 0) || !has_active_orders || had_fills_prev_step_;
      if (forced_requote) {
          action.should_requote = 1.0;  // Force requote
      }
      
      // Only requote if should_requote > 0, otherwise continue with existing quotes
      if (action.should_requote > 0.0) {
          adaptor_ptr->quote(action);
      }
      
      // Get trade count before advancing time to detect new fills
      double trade_count_before = strategy_ptr->getPosition().getNumberOfTrades();
      
      // Process time advancement and state updates
      // This advances ticks_per_step ticks, during which:
      // - Orders may fill (detected by execute() in SimExchange)
      // - Fills are processed by strategy.next() which calls position.onFill()
      isDone = !adaptor_ptr->next();
      ++steps;
      
      // Detect if fills occurred during this step by checking trade count change
      double trade_count_after = strategy_ptr->getPosition().getNumberOfTrades();
      had_fills_prev_step_ = (trade_count_after > trade_count_before);
      
      // Cache terminal info BEFORE reset can happen
      // This info will be exposed as final_info:* when done=true
      if (isDone) {
          adaptor_ptr->getInfo(terminal_info_);
          has_terminal_info_ = true;
      }
      
      WriteState();
  }

  void WriteState() {
    std::array<double, RLTrader::OBS_DIM> data;
    adaptor_ptr->getState(data);
    State state = Allocate(1);
    
    std::unordered_map<std::string, double> info;
    adaptor_ptr->getInfo(info);
    state["info:mid_price"_] = info["mid_price"];
    state["info:balance"_] = info["balance"];
    state["info:unrealized_pnl"_] = info["unrealized_pnl"];
    state["info:realized_pnl"_] = info["realized_pnl"];
    state["info:leverage"_] = info["leverage"];
    state["info:trade_count"_] = info["trade_count"];
    state["info:buy_trades"_] = info["buy_trades"];
    state["info:sell_trades"_] = info["sell_trades"];
    state["info:buy_amount"_] = info["buy_amount"];
    state["info:sell_amount"_] = info["sell_amount"];
    state["info:drawdown"_] = info["drawdown"];
    state["info:fees"_] = info["fees"];
    state["info:mid_diff"_] = info["mid_diff"];
    state["info:done"_] = isDone;
    state["info:net_position_usd"_] = info["net_position_usd"];
    state["info:net_amount_btc"_] = info["net_amount_btc"];
    state["info:last_bid_price"_] = info["last_bid_price"];
    state["info:last_ask_price"_] = info["last_ask_price"];
    state["info:last_mid_price"_] = info["last_mid_price"];
    
    // Expose terminal info from completed episode (for episode logging)
    // This is populated when isDone becomes true, before auto-reset
    if (has_terminal_info_) {
        state["info:final_realized_pnl"_] = terminal_info_["realized_pnl"];
        state["info:final_unrealized_pnl"_] = terminal_info_["unrealized_pnl"];
        state["info:final_trade_count"_] = terminal_info_["trade_count"];
        state["info:final_fees"_] = terminal_info_["fees"];
        state["info:final_net_amount_btc"_] = terminal_info_["net_amount_btc"];
        // Clear after use (will be repopulated when next episode ends)
        has_terminal_info_ = false;
        terminal_info_.clear();
    } else {
        state["info:final_realized_pnl"_] = 0.0;
        state["info:final_unrealized_pnl"_] = 0.0;
        state["info:final_trade_count"_] = 0.0;
        state["info:final_fees"_] = 0.0;
        state["info:final_net_amount_btc"_] = 0.0;
    }
    
    // === Reward Calculation with Rolling MA Anchor ===
    double mid_price = info["mid_price"];
    double leverage = info["leverage"];
    
    // 1. Initialize or update price anchor (slow-moving MA = "fair value")
    if (!price_anchor_initialized && mid_price > 0) {
        price_anchor_ma = mid_price;
        price_anchor_initialized = true;
    } else if (mid_price > 0) {
        price_anchor_ma = PRICE_MA_ALPHA * mid_price + (1.0 - PRICE_MA_ALPHA) * price_anchor_ma;
    }
    
    // 2. Realized PnL delta (clean, from closed trades)
    double realized_delta = info["realized_pnl"] - prev_realized_pnl;
    prev_realized_pnl = info["realized_pnl"];
    
    // 3. Unrealized PnL relative to rolling anchor (using INITIAL balance for consistent scaling)
    //    U = leverage * initial_balance * (price - anchor) / anchor
    //    Since leverage ≈ position_value / initial_balance, this gives:
    //    U ≈ position_value * price_deviation (consistent units)
    //    - Long (leverage > 0) when price > anchor → positive reward
    //    - Short (leverage < 0) when price < anchor → positive reward
    double unrealized_from_anchor = 0.0;
    if (price_anchor_ma > 0 && initial_balance_ > 0) {
        double price_deviation = (mid_price - price_anchor_ma) / price_anchor_ma;
        unrealized_from_anchor = leverage * initial_balance_ * price_deviation;
    }
    double unrealized_delta = unrealized_from_anchor - prev_unrealized_from_anchor;
    prev_unrealized_from_anchor = unrealized_from_anchor;
    
    // Fee rebate reward: with maker_fee < 0, fees becomes more negative when trading
    // We reward the agent for earning rebates (flip sign so rebates = positive reward)
    double current_fees = info["fees"];
    double fee_delta = -(current_fees - prev_fees);  // Positive when rebates earned
    prev_fees = current_fees;
    
    // Total reward: realized PnL delta + unrealized PnL delta + fee rebates
    // Agent now gets rewarded for trading (via maker rebates)
    double reward = realized_delta + unrealized_delta + fee_delta;
    state["reward"_] = reward;
    
    state["obs"_].Assign(data.begin(), data.size());
  }

  bool IsDone() override { return isDone; }
};

using RlTraderLitePool = AsyncLitePool<RlTraderEnv>;

}  // namespace rltrader

#endif  // LITEPOOL_RLTRADER_RLTRADER_LITEPOOL_H_
