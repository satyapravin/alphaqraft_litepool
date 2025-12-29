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

// Soft clipping function that preserves gradients
// Linear within [-threshold, threshold], soft beyond
inline double softsign_clip(double x, double threshold = 1.0) {
    double scale = threshold;
    return scale * (x / scale) / (1.0 + std::abs(x / scale));
}

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
    return MakeDict("obs"_.Bind(Spec<double>({RLTrader::OBS_DIM})),  // 13 market + 4 AMM + 8 trade + 11 agent state + 1 previous spread + 2 bid/ask distances + 1 mid_change = 40
                    "info:mid_price"_.Bind(Spec<double>({-1})),
                    "info:balance"_.Bind(Spec<double>({-1})),
                    "info:unrealized_pnl"_.Bind(Spec<double>({-1})),
                    "info:lifo_unrealized_pnl"_.Bind(Spec<double>({-1})),  // LIFO unrealized (consistent with spread_capture)
                    "info:realized_pnl"_.Bind(Spec<double>({-1})),
                    "info:leverage"_.Bind(Spec<double>({-1})),
                    "info:target_inventory"_.Bind(Spec<double>({-1})),  // Agent's desired inventory level (EMA smoothed)
                    "info:risk_aversion"_.Bind(Spec<double>({-1})),  // Risk aversion parameter (γ) for A-S model
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
                    "info:market_bid_price"_.Bind(Spec<double>({-1})),
                    "info:market_ask_price"_.Bind(Spec<double>({-1})),
                    "info:net_amount_btc_raw"_.Bind(Spec<double>({-1})),
                    "info:average_price_raw"_.Bind(Spec<double>({-1})),
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
                    "info:final_net_amount_btc"_.Bind(Spec<double>({-1})),
                    "info:final_spread_capture"_.Bind(Spec<double>({-1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // 4-action space: bid_spread, ask_spread, target_inventory, risk_aversion
    // Note: requote removed - we use smart requote (only when prices change by >5 ticks)
    return MakeDict("action"_.Bind(Spec<float>({4}, {{  0.,  0., -1.,  0.0 },
                                                     {  1.,  1.,  1.,  1.0 }})));
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
  double prev_realized_pnl = 0.0;         // For logging only
  double prev_unrealized_pnl = 0.0;       // Weighted-average unrealized (for logging, matches balance cash flow)
  double prev_lifo_unrealized_pnl = 0.0;  // LIFO unrealized (for inv_reward, consistent with spread_capture)
  double prev_fees = 0.0;                 // For mm_reward (fee rebates)
  double prev_spread_capture = 0.0;       // For mm_reward (LIFO round-trip profit)
  double initial_balance_ = 0.0;          // Store initial balance for consistent reward scaling
  double prev_flow_misalignment_ = 0.0;   // Flow misalignment tracking for delta penalty
  
  // Terminal info cache (stores metrics before reset for episode logging)
  std::unordered_map<std::string, double> terminal_info_;
  bool has_terminal_info_ = false;
  
  // Track fills from previous step to force requote if orders were filled
  bool had_fills_prev_step_ = false;
  
  // Track last quoted prices - only requote if prices change significantly
  double prev_quoted_bid_ = 0.0;
  double prev_quoted_ask_ = 0.0;
  // Price change threshold: 2 ticks minimum to trigger requote
  // Lower threshold = quotes track market better, more fills
  static constexpr double REQUOTE_TICK_THRESHOLD = 2.0;

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
        ResetInternal();
  }
  
  void ResetInternal() {
    // Reset can be called when:
    // 1. Episode is done (isDone == true) - normal case
    // 2. Force reset (force_reset == true) - can happen at any time
    // So we don't check isDone here - it's valid to reset even if not done
    
    prev_realized_pnl = 0.0;
    prev_unrealized_pnl = 0.0;
    prev_lifo_unrealized_pnl = 0.0;
    prev_fees = 0.0;
    prev_spread_capture = 0.0;
    initial_balance_ = balance;  // Store initial balance for consistent reward scaling
    prev_flow_misalignment_ = 0.0;  // Reset flow misalignment tracking
    steps = 0;
    had_fills_prev_step_ = false;  // Reset fill tracking
    prev_quoted_bid_ = 0.0;       // Reset quote tracking
    prev_quoted_ask_ = 0.0;
    
    // Track if any reset step fails - we still need to call WriteState!
    bool reset_failed = false;
    long long book_start_timestamp = 0;
    
    // Reset exchange first (picks random starting row)
        exchange_ptr->reset();
    
    // Get starting timestamp from book reader to sync trade reader
    if (!reset_failed) {
        RLTrader::SimExchange* sim_exch = dynamic_cast<RLTrader::SimExchange*>(exchange_ptr.get());
        if (sim_exch) {
                // Peek at first timestamp without consuming the row
                // CRITICAL: hasData() checks if dataReader.hasNext(), which ensures rows is populated
                // Only call peekFirstTimestamp() if hasData() returns true
                if (sim_exch->hasData()) {
                    book_start_timestamp = sim_exch->peekFirstTimestamp();
                } else {
                    // No data available after reset - this should not happen if CSV has enough data
                reset_failed = true;
            }
        }
    }
    
    // Reset adaptor (which will reset trade reader if present)
    if (!reset_failed) {
            adaptor_ptr->reset();
            
            // Sync trade reader to book's starting timestamp
            if (book_start_timestamp > 0) {
                adaptor_ptr->syncTradeReader(book_start_timestamp);
            }
            
            isDone = false;
    }
    
    // CRITICAL: ALWAYS call WriteState at end of Reset
    // WriteState() calls Allocate() first, ensuring done_write() works
    // This prevents deadlock regardless of what failed above
    WriteState();
  }

  void Step(const Action& action_dict) override { 
          StepInternal(action_dict);
  }
  
  void StepInternal(const Action& action_dict) {
      // Timing for performance measurement
      static thread_local int step_count = 0;
      static thread_local double total_cpp_time = 0.0;
      
      auto step_start = std::chrono::high_resolution_clock::now();
      
      ++step_count;  // Keep counter for potential future use 
      RLTrader::RLAction action;
      // 4-action space: bid_spread, ask_spread, target_inventory, risk_aversion
      // Requote is handled automatically (only when prices change by >5 ticks)
      action.bid_spread       = static_cast<double>(action_dict["action"_][0]);
      action.ask_spread       = static_cast<double>(action_dict["action"_][1]);
      action.target_inventory = static_cast<double>(action_dict["action"_][2]);
      action.risk_aversion    = static_cast<double>(action_dict["action"_][3]);
      action.should_requote   = 0.0;  // Not used - smart requote logic handles this
      
      // Update target inventory and risk aversion (direct assignment, no smoothing)
      strategy_ptr->updateTargetInventory(action.target_inventory, action.risk_aversion);
      
      // === SMART REQUOTE LOGIC ===
      // Requote is handled automatically (no agent action) to prevent gaming.
      // We only requote when:
      // 1. First step (no orders exist after reset)
      // 2. No active orders in the market
      // 3. Previous step had fills (need to replace filled orders)
      // 4. Proposed quote prices differ from current quotes by more than 5 ticks
      //
      // This reduces order churn while still allowing price adjustments.
      
      bool has_active_orders = !exchange_ptr->getBidOrders().empty() || 
                               !exchange_ptr->getAskOrders().empty();
      bool forced_requote = (steps == 0) || !has_active_orders || had_fills_prev_step_;
      
      // Check if prices changed enough to warrant requote (2 tick threshold)
      bool prices_changed = adaptor_ptr->shouldRequote(action, REQUOTE_TICK_THRESHOLD);
      
      // Requote if forced OR prices changed significantly
      bool should_requote = forced_requote || prices_changed;
      
      if (should_requote) {
          adaptor_ptr->quote(action);
          // Update tracked quote prices
          prev_quoted_bid_ = strategy_ptr->getLastBidPrice();
          prev_quoted_ask_ = strategy_ptr->getLastAskPrice();
      }
      
      // Get trade count before advancing time to detect new fills
      double trade_count_before = strategy_ptr->getPosition().getNumberOfTrades();
      
      // Advance time and read data
      bool has_data = adaptor_ptr->next();
      
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
        adaptor_ptr->getState(data);
    
    std::unordered_map<std::string, double> info;
    adaptor_ptr->getInfo(info);
    
    state["info:mid_price"_] = info["mid_price"];
    state["info:balance"_] = info["balance"];
    state["info:unrealized_pnl"_] = info["unrealized_pnl"];
    state["info:lifo_unrealized_pnl"_] = info["lifo_unrealized_pnl"];
    state["info:realized_pnl"_] = info["realized_pnl"];
    state["info:leverage"_] = info["leverage"];
    state["info:target_inventory"_] = info["target_inventory"];  // Agent's desired inventory level (EMA smoothed)
    state["info:risk_aversion"_] = info["risk_aversion"];  // Risk aversion parameter (γ) for A-S model
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
    state["info:market_bid_price"_] = info["market_bid_price"];
    state["info:market_ask_price"_] = info["market_ask_price"];
    state["info:net_amount_btc_raw"_] = info["net_amount_btc_raw"];
    state["info:average_price_raw"_] = info["average_price_raw"];
    
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
        state["info:final_unrealized_pnl"_] = terminal_info_["lifo_unrealized_pnl"];  // Use LIFO for consistency
        state["info:final_trade_count"_] = terminal_info_["trade_count"];
        state["info:final_fees"_] = terminal_info_["fees"];
        state["info:final_net_amount_btc"_] = terminal_info_["net_amount_btc"];
        state["info:final_spread_capture"_] = terminal_info_["spread_capture"];
        // Clear after use (will be repopulated when next episode ends)
        has_terminal_info_ = false;
        terminal_info_.clear();
    } else {
        state["info:final_realized_pnl"_] = 0.0;
        state["info:final_unrealized_pnl"_] = 0.0;
        state["info:final_trade_count"_] = 0.0;
        state["info:final_fees"_] = 0.0;
        state["info:final_net_amount_btc"_] = 0.0;
        state["info:final_spread_capture"_] = 0.0;
    }
    
    // === Reward Calculation for Hierarchical RL ===
    // Two separate reward streams for two agents:
    // - MM Agent: spread capture (LIFO) + fee rebates (execution quality)
    // - Inventory Agent: unrealized P&L delta (market direction)
    //
    // MM agent optimizes for completing profitable round-trips + earning rebates
    // Inv agent optimizes for holding inventory in the right direction
    
    // Reward scaling: Use log rewards to handle wide dynamic range
    // Log rewards: sign(r) * log(1 + |r|) compresses large rewards while preserving sign
    // This prevents extreme advantage values and stabilizes learning
    constexpr double MM_REWARD_SCALE = 10000.0;  // Scale before log transform
    constexpr double INV_REWARD_SCALE = 10000.0; // Scale before log transform (increased from 100.0 to match MM scale)
    constexpr bool USE_LOG_REWARDS = true;        // Use log transform for reward normalization
    
    // 1. Spread capture delta (LIFO profit from round-trips)
    double current_spread_capture = info["spread_capture"];
    double spread_capture_delta = current_spread_capture - prev_spread_capture;
    prev_spread_capture = current_spread_capture;
    if (initial_balance_ > 1e-9) {
        spread_capture_delta /= initial_balance_;
    } else {
        spread_capture_delta = 0.0;
    }
    
    // 2. Fee delta - positive when rebates earned (maker fees are negative)
    double current_fees = info["fees"];
    double fee_delta = -(current_fees - prev_fees);
    prev_fees = current_fees;
    if (initial_balance_ > 1e-9) {
        fee_delta /= initial_balance_;
    } else {
        fee_delta = 0.0;
    }
    
    // 3. LIFO Unrealized P&L delta (for inv_reward)
    // Use LIFO unrealized to be consistent with spread_capture accounting
    // This ensures: spread_capture + lifo_unrealized = total P&L
    double current_lifo_unrealized = info["lifo_unrealized_pnl"];
    double lifo_unrealized_delta = current_lifo_unrealized - prev_lifo_unrealized_pnl;
    
    // CRITICAL: On first step after reset, prev_lifo_unrealized_pnl is 0, so delta = current value
    // This can cause a huge first-step reward if position is already underwater
    // Skip first step delta to prevent this (steps == 0 means first step after reset)
    if (steps == 0) {
        lifo_unrealized_delta = 0.0;  // Skip first step to prevent huge initial delta
    }
    
    prev_lifo_unrealized_pnl = current_lifo_unrealized;
    if (initial_balance_ > 1e-9) {
        lifo_unrealized_delta /= initial_balance_;
    } else {
        lifo_unrealized_delta = 0.0;
    }
    
    // Also track weighted-average unrealized for display/penalties (not used in reward)
    // This is weighted-average (matches balance cash flow), not LIFO
    double current_unrealized_pnl = info["unrealized_pnl"];
    prev_unrealized_pnl = current_unrealized_pnl;
    
    // Track realized P&L for logging only (not used in reward)
    // This is weighted-average realized PnL (matches balance cash flow), not LIFO spreadCapture
    double current_realized_pnl = info["realized_pnl"];
    prev_realized_pnl = current_realized_pnl;
    
    // === MM Agent Reward: execution quality ===
    // spread_capture: LIFO profit from completing round-trips
    // fee_delta: maker rebates earned from providing liquidity
    // Both are directly controllable by the agent's quoting behavior
    double mm_reward_raw = (spread_capture_delta + fee_delta) * MM_REWARD_SCALE;
    
    // Apply log transform if enabled: sign(r) * log(1 + |r|)
    // This compresses large rewards while preserving sign and relative ordering
    double mm_reward = USE_LOG_REWARDS 
        ? (mm_reward_raw >= 0 ? 1.0 : -1.0) * std::log(1.0 + std::abs(mm_reward_raw))
        : mm_reward_raw;
    
    // === Inventory Agent Reward: market direction ===
    // Rewards holding inventory in the right direction (total P&L = realized + unrealized)
    // Total P&L delta = spread_capture_delta + lifo_unrealized_delta
    // This gives the inventory agent the full picture of market direction
    // The inventory agent controls WHAT position to hold, so it should see the total impact
    double total_pnl_delta = spread_capture_delta + lifo_unrealized_delta;
    double inv_reward_raw = total_pnl_delta * INV_REWARD_SCALE;
    
    // Apply log transform if enabled
    // CRITICAL: Ensure log transform is actually applied to prevent extreme values
    // If inv_reward_raw is -15700, log transform should give -log(1+15700) ≈ -9.66, not -15700
    // Use natural log (log) not log10 - log10 compresses 2.3x less, leading to larger values
    double inv_reward_instant;
    if constexpr (USE_LOG_REWARDS) {
        // Use constexpr if to ensure compile-time evaluation and prevent optimization issues
        // Natural log provides better compression: log(10001) ≈ 9.21 vs log10(10001) ≈ 4.0
        inv_reward_instant = (inv_reward_raw >= 0 ? 1.0 : -1.0) * std::log(1.0 + std::abs(inv_reward_raw));
    } else {
        inv_reward_instant = inv_reward_raw;
    }
    
    // No EMA smoothing needed - inventory agent updates every step now
    double inv_reward = inv_reward_instant;
   
    // Cumulative flow alignment penalty: guide inventory agent to follow flow direction
    double cumulative_flow_signal = data[16];  // Normalized cumulative flow [-1, 1] from observation
    double target_inventory_signal = info["target_inventory"];  // Current target inventory (EMA smoothed)
    
    // Calculate misalignment: penalty when signs are opposite
    constexpr double TYPICAL_TARGET_RANGE = 2.0;  // Match config.target_range
    double target_normalized = target_inventory_signal / TYPICAL_TARGET_RANGE;  // Normalize to [-1, 1] range
    double flow_alignment = cumulative_flow_signal * target_normalized;  // Product in [-1, 1]
    double flow_misalignment = std::max(0.0, -flow_alignment);  // Positive when misaligned (opposite signs), in [0, 1]
    
    // Calculate DELTA: only penalize when misalignment INCREASES (agent is getting worse)
    double flow_misalignment_delta = flow_misalignment - prev_flow_misalignment_;
    prev_flow_misalignment_ = flow_misalignment;
    
    // Only penalize increases in misalignment (positive delta)
    double misalignment_increase = std::max(0.0, flow_misalignment_delta);
    
    // Scale penalty: use very small scale since this is a delta (not persistent)
    constexpr double FLOW_ALIGNMENT_PENALTY_SCALE = 10.0;  // Scale for delta penalty
    double flow_penalty_raw = -misalignment_increase * FLOW_ALIGNMENT_PENALTY_SCALE;
    double flow_alignment_penalty = USE_LOG_REWARDS
        ? (flow_penalty_raw >= 0 ? 1.0 : -1.0) * std::log(1.0 + std::abs(flow_penalty_raw))
        : flow_penalty_raw;

    // Apply penalties to inventory reward
    double inv_reward_with_penalties = inv_reward + flow_alignment_penalty;
    
    // Soft clip final rewards to prevent extreme values from dominating learning
    // Soft clipping preserves gradients better than hard clipping
    // Threshold of 2.0: linear within [-2, 2], soft beyond
    double mm_reward_clipped = softsign_clip(mm_reward, 2.0);
    double inv_reward_clipped = softsign_clip(inv_reward_with_penalties, 2.0);
    
    state["info:mm_reward"_] = mm_reward_clipped;
    state["info:inv_reward"_] = inv_reward_clipped;
    
    // === Combined Reward: for backwards compatibility ===
    // Use clipped rewards for combined reward (sum can exceed [-2, 2] but that's fine for combined)
    double reward = mm_reward_clipped + inv_reward_clipped;
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
