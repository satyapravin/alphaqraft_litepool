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
#include <chrono>
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
                    "ticks_per_step"_.Bind<int>(5),  // Advance 5 ticks per RL step
                    "base_spread_bps"_.Bind<double>(1.0),  // Base spread in basis points
                    "min_size_pct"_.Bind<double>(0.5),      // Minimum order size as % of balance
                    "max_size_pct"_.Bind<double>(2.0));    // Maximum order size as % of balance
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<double>({32})),  // 13 market + 4 AMM flow + 8 trade + 7 agent state
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
                    "info:deviation_from_target"_.Bind(Spec<double>({-1})),
                    "info:spread_capture"_.Bind(Spec<double>({-1})),
                    // Hierarchical RL: separate reward streams for two agents
                    "info:mm_reward"_.Bind(Spec<double>({-1})),   // MM agent: realized + spread_capture + fees
                    "info:inv_reward"_.Bind(Spec<double>({-1})),  // Inventory agent: unrealized P&L delta
                    // Terminal info from completed episode (available after auto-reset)
                    "info:final_realized_pnl"_.Bind(Spec<double>({-1})),
                    "info:final_unrealized_pnl"_.Bind(Spec<double>({-1})),
                    "info:final_trade_count"_.Bind(Spec<double>({-1})),
                    "info:final_fees"_.Bind(Spec<double>({-1})),
                    "info:final_net_amount_btc"_.Bind(Spec<double>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // 4-action space: bid_spread, ask_spread, target_inventory, should_requote
    // Note: skew removed - automatically computed from inventory error toward target
    return MakeDict("action"_.Bind(Spec<float>({4}, {{ -1., -1., -1., -1. },
                                                     {  1.,  1.,  1.,  1. }})));
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
  int ticks_per_step = 5;
  double base_spread_bps = 0.0;
  double min_size_pct = 0.5;
  double max_size_pct = 2.0;
  long long steps = 0;
  
  // Reward tracking (used for hierarchical RL reward streams)
  double prev_realized_pnl = 0.0;    // For mm_reward
  double prev_unrealized_pnl = 0.0;  // For inv_reward
  double prev_fees = 0.0;            // For mm_reward (fee rebates)
  double initial_balance_ = 0.0; // Store initial balance for consistent reward scaling
  
  // Terminal info cache (stores metrics before reset for episode logging)
  std::unordered_map<std::string, double> terminal_info_;
  bool has_terminal_info_ = false;
  
  // Track fills from previous step to force requote if orders were filled
  bool had_fills_prev_step_ = false;
  
  // Track if RL chose to requote voluntarily (not forced by auto-requote logic)
  // Used to penalize excessive voluntary requoting
  bool rl_chose_requote_ = false;

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


    std::string trade_filename;
    if (this->is_prod) {
      exch_raw_ptr = new RLTrader::CryptoExchange(symbol, hedge_symbol, api_key, api_secret);
      // No trade file for production (live exchange)
      trade_filename = "";
    } else {
      int idx = env_id % 64;
      // New folder structure: foldername/books/1.csv and foldername/trades/1.csv
      // Try .csv first (uncompressed), then fallback to .csv.gz if needed
      std::string book_filename = foldername + "books/" + std::to_string(idx + 1) + ".csv";
      trade_filename = foldername + "trades/" + std::to_string(idx + 1) + ".csv";
      
      // Note: Files are expected to be uncompressed .csv after processing
      // If .csv doesn't exist, user can provide .csv.gz files (CsvReader handles .gz if compiled with zlib)
      exch_raw_ptr = new RLTrader::SimExchange(book_filename, 250, start_read);
    }

    instr_ptr.reset(instr_raw_ptr);
    exchange_ptr.reset(exch_raw_ptr);
    RLTrader::StrategyConfig config;  // Use default config
    config.base_spread_bps = base_spread_bps;  // From Python config
    config.min_size_pct = min_size_pct;        // From Python config
    config.max_size_pct = max_size_pct;        // From Python config
    strategy_ptr = std::make_unique<RLTrader::Strategy>(*instr_ptr, *exchange_ptr, balance, 20, config);
    adaptor_ptr = std::make_unique<RLTrader::EnvAdaptor>(*strategy_ptr, *exchange_ptr, trade_filename, ticks_per_step);
  }

  void Reset() override {
    // CRITICAL: Wrap entire Reset in try-catch to ensure WriteState() is ALWAYS called
    // Without this, any uncaught exception causes deadlock (Python waits forever for state)
    try {
        ResetInternal();
    } catch (...) {
        // If anything goes wrong, still write state to prevent deadlock
        isDone = true;
        // WriteState() MUST be called - it calls Allocate() which sets up the semaphore callback
        // Even if WriteState() throws, Allocate() calls Done(1) before throwing, signaling semaphore
        WriteState();
    }
  }
  
  void ResetInternal() {
    // Reset can be called when:
    // 1. Episode is done (isDone == true) - normal case
    // 2. Force reset (force_reset == true) - can happen at any time
    // So we don't check isDone here - it's valid to reset even if not done
    
    prev_realized_pnl = 0.0;
    prev_unrealized_pnl = 0.0;
    prev_fees = 0.0;
    initial_balance_ = balance;  // Store initial balance for consistent reward scaling
    steps = 0;
    had_fills_prev_step_ = false;  // Reset fill tracking
    rl_chose_requote_ = false;    // Reset requote tracking
    
    // Track if any reset step fails - we still need to call WriteState!
    bool reset_failed = false;
    long long book_start_timestamp = 0;
    
    // Reset exchange first (picks random starting row)
    try {
        exchange_ptr->reset();
    } catch (...) {  // Catch ALL exceptions, not just std::exception
        reset_failed = true;
    }
    
    // Get starting timestamp from book reader to sync trade reader
    if (!reset_failed) {
        RLTrader::SimExchange* sim_exch = dynamic_cast<RLTrader::SimExchange*>(exchange_ptr.get());
        if (sim_exch) {
            try {
                // Peek at first timestamp without consuming the row
                // CRITICAL: hasData() checks if dataReader.hasNext(), which ensures rows is populated
                // Only call peekFirstTimestamp() if hasData() returns true
                if (sim_exch->hasData()) {
                    book_start_timestamp = sim_exch->peekFirstTimestamp();
                } else {
                    // No data available after reset - this should not happen if CSV has enough data
                    reset_failed = true;
                }
            } catch (const std::exception& e) {
                // If peekFirstTimestamp() throws (e.g., rows is empty), mark reset as failed
                reset_failed = true;
            } catch (...) {  // Catch ALL other exceptions
                reset_failed = true;
            }
        }
    }
    
    // Reset adaptor (which will reset trade reader if present)
    if (!reset_failed) {
        try {
            adaptor_ptr->reset();
            
            // Sync trade reader to book's starting timestamp
            if (book_start_timestamp > 0) {
                adaptor_ptr->syncTradeReader(book_start_timestamp);
            }
            
            isDone = false;
        } catch (...) {
            reset_failed = true;
        }
    }
    
    // If any reset step failed, mark episode as done
    if (reset_failed) {
        isDone = true;
    }
    
    // CRITICAL: ALWAYS call WriteState at end of Reset
    // WriteState() calls Allocate() first, ensuring done_write() works
    // This prevents deadlock regardless of what failed above
    WriteState();
  }

  void Step(const Action& action_dict) override { 
      // CRITICAL: Wrap entire Step in try-catch to ensure WriteState() is ALWAYS called
      // Without this, any uncaught exception causes deadlock (Python waits forever for state)
      try {
          StepInternal(action_dict);
      } catch (...) {
          // If anything goes wrong, still write state to prevent deadlock
          isDone = true;
          WriteState();
      }
  }
  
  void StepInternal(const Action& action_dict) {
      // Timing for performance measurement
      static thread_local int step_count = 0;
      static thread_local double total_cpp_time = 0.0;
      
      auto step_start = std::chrono::high_resolution_clock::now();
      
      ++step_count;  // Keep counter for potential future use 
      RLTrader::RLAction action;
      // 4-action space: bid_spread, ask_spread, target_inventory, should_requote
      // Note: skew removed - automatically computed from inventory error
      action.bid_spread       = static_cast<double>(action_dict["action"_][0]);
      action.ask_spread       = static_cast<double>(action_dict["action"_][1]);
      action.target_inventory = static_cast<double>(action_dict["action"_][2]);
      action.should_requote   = static_cast<double>(action_dict["action"_][3]);
      
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
      
      // Track if RL voluntarily chose to requote (for penalty)
      // Voluntary = agent chose requote AND it wasn't forced
      rl_chose_requote_ = (action.should_requote > 0.0) && !forced_requote;
      
      // Only requote if should_requote > 0, otherwise continue with existing quotes
      if (action.should_requote > 0.0) {
      adaptor_ptr->quote(action);
      }
      
      // Get trade count before advancing time to detect new fills
      double trade_count_before = strategy_ptr->getPosition().getNumberOfTrades();
      
      // Advance time and read data
      bool has_data = false;
      try {
          has_data = adaptor_ptr->next();
      } catch (...) {
          has_data = false;
      }
      
      // CRITICAL: current_step_ is incremented in PreProcess() BEFORE Step() is called.
      // So when WriteState() calls Allocate(), current_step_ represents the step we're
      // currently processing (the step number for this Step() call).
      // We increment steps AFTER reading data, so steps represents steps completed.
      int max_episode_steps = spec_.config["max_episode_steps"_];
      
      // Increment step count (we attempted/completed this step)
      ++steps;
      
      // CRITICAL: Episode MUST end when steps >= max_episode_steps.
      // The truncation check in Allocate() uses: trunc = done && (current_step_ >= max_episode_steps)
      // Since PreProcess() increments current_step_ before Step(), when we process step N:
      // - PreProcess() sets current_step_ = N
      // - Step() increments steps, so if steps was N-1, it becomes N
      // - When WriteState() calls Allocate(), current_step_ = N and steps = N
      // So when steps = max_episode_steps, current_step_ should also be max_episode_steps.
      // We MUST set isDone = true unconditionally when steps >= max_episode_steps.
      // 
      // IMPORTANT: Check steps >= max_episode_steps FIRST, before checking has_data.
      // This ensures that if we've reached max steps, we always truncate, even if has_data is false.
      // If has_data is false but steps < max_episode_steps, that's early termination (not truncated).
      if (steps >= max_episode_steps) {
          // Reached max episode steps - episode ends (truncated)
          // CRITICAL: Always set isDone = true when steps >= max_episode_steps, regardless of has_data
          isDone = true;
      } else if (!has_data) {
          // No more data - must end episode (can't advance without CSV rows)
          // This is early termination, not truncation (trunc will be false because current_step_ < max_episode_steps)
          isDone = true;
      } else {
          // Continue episode
          isDone = false;
      }
      
      // SAFEGUARD: Double-check that isDone is set correctly when we've reached max steps
      // This prevents the bug where env 4 continues past 2048 steps
      // This should never trigger if the logic above is correct, but it's a safety net
      if (steps >= max_episode_steps && !isDone) {
          // Force isDone = true - this should never happen but prevents infinite loops
          isDone = true;
      }
      
      // CRITICAL: Ensure isDone is ALWAYS true when steps >= max_episode_steps
      // This is the ultimate safeguard - even if the logic above somehow fails,
      // we MUST end the episode when we've exceeded max_episode_steps.
      // This prevents environments from running indefinitely (like env 4 reaching 4095 steps).
      // 
      // IMPORTANT: This check MUST happen BEFORE WriteState() is called, because
      // WriteState() calls Allocate() which calls IsDone() to determine done/trunc.
      // If isDone is false when Allocate() is called, it will set done=false and trunc=false,
      // and our override in WriteState() might not work correctly.
      if (steps >= max_episode_steps) {
          isDone = true;
      }
      
      // Detect if fills occurred during this step by checking trade count change
      double trade_count_after = strategy_ptr->getPosition().getNumberOfTrades();
      had_fills_prev_step_ = (trade_count_after > trade_count_before);
      
      // Cache terminal info BEFORE reset can happen
      // This info will be exposed as final_info:* when done=true
      if (isDone) {
          adaptor_ptr->getInfo(terminal_info_);
          has_terminal_info_ = true;
      }
      
      // FINAL SAFEGUARD: Double-check isDone one more time before WriteState()
      // This ensures that even if something reset isDone above, we catch it here.
      // This is the last chance to fix isDone before Allocate() is called.
      if (steps >= max_episode_steps && !isDone) {
          // This should never happen, but if it does, force isDone = true
          // This is critical to prevent infinite loops
          isDone = true;
      }
      
      WriteState();
      
      // Measure C++ processing time at end of Step() (includes WriteState())
      auto step_end = std::chrono::high_resolution_clock::now();
      double step_time = std::chrono::duration<double>(step_end - step_start).count();
      total_cpp_time += step_time;
  }

  void WriteState() {
    // Allocate state buffer - base class Allocate() calls IsDone() which returns
    // true when steps >= max_episode_steps, so done will be set correctly
    State state = Allocate(1);
    
    std::array<double, RLTrader::OBS_DIM> data;
    try {
        adaptor_ptr->getState(data);
    } catch (...) {
        // If getState fails, fill with zeros and mark done
        data.fill(0.0);
        isDone = true;
    }
    
    std::unordered_map<std::string, double> info;
    try {
    adaptor_ptr->getInfo(info);
    } catch (...) {
        // If getInfo fails, use empty info (all zeros)
        isDone = true;
    }
    
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
    
    // Leverage deviation from target (for monitoring and debugging)
    // Positive when leverage > target (over-leveraged), negative when leverage < target (under-leveraged)
    double leverage = info["leverage"];  // Get leverage from info
    double target_leverage_info = strategy_ptr->getTargetInventory();
    double deviation_from_target = leverage - target_leverage_info;
    state["info:deviation_from_target"_] = deviation_from_target;
    state["info:spread_capture"_] = info["spread_capture"];
    
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
    
    // === Reward Calculation for Hierarchical RL ===
    // Two separate reward streams for two agents:
    // - MM Agent: realized P&L + fees (execution quality)
    // - Inventory Agent: unrealized P&L delta (market direction)
    //
    // The combined "reward" = mm_reward + inv_reward = total wealth change
    // This matches the episode logs: Net = R.PnL + U.PnL + Fees
    
    constexpr double REWARD_SCALE = 10000.0;  // Scale for readability
    
    // 1. Realized P&L delta (for mm_reward)
    double current_realized_pnl = info["realized_pnl"];
    double realized_pnl_delta = current_realized_pnl - prev_realized_pnl;
    prev_realized_pnl = current_realized_pnl;
    if (initial_balance_ > 1e-9) {
        realized_pnl_delta /= initial_balance_;
    } else {
        realized_pnl_delta = 0.0;
    }
    
    // 2. Fee delta (for mm_reward) - positive when rebates earned
    double current_fees = info["fees"];
    double fee_delta = -(current_fees - prev_fees);
    prev_fees = current_fees;
    if (initial_balance_ > 1e-9) {
        fee_delta /= initial_balance_;
    } else {
        fee_delta = 0.0;
    }
    
    // 3. Unrealized P&L delta (for inv_reward)
    double current_unrealized_pnl = info["unrealized_pnl"];
    double unrealized_pnl_delta = current_unrealized_pnl - prev_unrealized_pnl;
    prev_unrealized_pnl = current_unrealized_pnl;
    if (initial_balance_ > 1e-9) {
        unrealized_pnl_delta /= initial_balance_;
    } else {
        unrealized_pnl_delta = 0.0;
    }
    
    // === MM Agent Reward: execution quality ===
    // Rewards closing positions profitably: realized P&L + fee rebates
    // Note: Scale doesn't need to match Inv reward - advantages are normalized independently
    double mm_reward = (realized_pnl_delta + fee_delta) * REWARD_SCALE;
    state["info:mm_reward"_] = mm_reward;
    
    // === Inventory Agent Reward: market direction ===
    // Rewards holding inventory in the right direction (unrealized P&L)
    // Matches episode log: U.PnL
    double inv_reward = unrealized_pnl_delta * REWARD_SCALE;
    state["info:inv_reward"_] = inv_reward;
    
    // === Combined Reward: total wealth change ===
    // mm_reward + inv_reward = R.PnL + Fees + U.PnL = Net
    double reward = mm_reward + inv_reward;
    state["reward"_] = reward;
    
    state["obs"_].Assign(data.begin(), data.size());
  }

  bool IsDone() override { 
      // CRITICAL: Always return true if steps >= max_episode_steps, regardless of isDone
      // This ensures that Allocate() will set done=true and trunc=true correctly.
      // This is the ultimate safeguard to prevent environments from running past max_episode_steps.
      int max_episode_steps = spec_.config["max_episode_steps"_];
      if (steps >= max_episode_steps) {
          return true;  // Force done when we've exceeded max steps
      }
      return isDone; 
  }
};

using RlTraderLitePool = AsyncLitePool<RlTraderEnv>;

}  // namespace rltrader

#endif  // LITEPOOL_RLTRADER_RLTRADER_LITEPOOL_H_
