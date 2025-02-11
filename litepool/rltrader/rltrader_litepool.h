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
#include <iostream>
#include "litepool/core/async_litepool.h"
#include "litepool/core/env.h"
#include "env_adaptor.h"

#include "base_instrument.h"
#include "inverse_instrument.h"
#include "normal_instrument.h"

#include <filesystem>

#include "deribit_exchange.h"
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
                    "tick_size"_.Bind<float>(0.5),
                    "min_amount"_.Bind<float>(10.0),
                    "maker_fee"_.Bind<float>(-0.0001),
                    "taker_fee"_.Bind<float>(0.0005),
                    "foldername"_.Bind(std::string("")),
                    "balance"_.Bind(1.0),
                    "depth"_.Bind<int>(5),
                    "start"_.Bind<int>(0),
                    "max"_.Bind<int>(72000));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<float>({490})),
                    "info:mid_price"_.Bind(Spec<float>({})),
                    "info:balance"_.Bind(Spec<float>({})),
                    "info:unrealized_pnl"_.Bind(Spec<float>({})),
                    "info:realized_pnl"_.Bind(Spec<float>({})),
                    "info:leverage"_.Bind(Spec<float>({})),
                    "info:trade_count"_.Bind(Spec<float>({})),
                    "info:inventory_drawdown"_.Bind(Spec<float>({})), // NOLINT(*-static-accessed-through-instance)
                    "info:drawdown"_.Bind(Spec<float>({})),
                    "info:fees"_.Bind((Spec<float>({}))),
                    "info:buy_amount"_.Bind((Spec<float>({}))),
                    "info:sell_amount"_.Bind((Spec<float>({}))));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({8}, {{-1., -1., -1., -1., -1., -1., -1., -1.},
                                                     { 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.}})));
  }
};


using RlTraderEnvSpec = EnvSpec<RlTraderEnvFns>;

class RlTraderEnv : public Env<RlTraderEnvSpec> {
 protected:
  int spreads[4] = {0, 4, 8, 16};
  int state_{0};
  bool isDone = true;
  bool is_prod = false;
  bool is_inverse_instr = false;
  std::string api_key;
  std::string api_secret;
  std::string symbol;
  double tick_size;
  double min_amount;
  double maker_fee;
  double taker_fee;
  std::string foldername;
  double balance = 0;
  int start_read = 0;
  int max_read = 0;
  long long steps = 0;
  double previous_rpnl = 0;
  double previous_upnl = 0;
  double previous_fees = 0;
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
                                              tick_size(spec.config["tick_size"_]),
                                              min_amount(spec.config["min_amount"_]),
                                              maker_fee(spec.config["maker_fee"_]),
                                              taker_fee(spec.config["taker_fee"_]),
                                              foldername(spec.config["foldername"_]),
                                              balance(spec.config["balance"_]),
                                              start_read(spec.config["start"_]),
                                              max_read(spec.config["max"_])
  {

    RLTrader::BaseInstrument* instr_raw_ptr = nullptr;
    RLTrader::BaseExchange* exch_raw_ptr = nullptr;


    if (this->is_inverse_instr) {
      instr_raw_ptr = new RLTrader::InverseInstrument(symbol, tick_size, min_amount, maker_fee, taker_fee);
    } else {
      instr_raw_ptr = new RLTrader::NormalInstrument(symbol, tick_size, min_amount, maker_fee, taker_fee);
    }


    if (this->is_prod) {
      exch_raw_ptr = new RLTrader::DeribitExchange(symbol, api_key, api_secret);
    } else {
      int idx = env_id % 45;
      std::string filename = foldername + std::to_string(idx + 1) + ".csv";
      std::cout << filename << std::endl;
      exch_raw_ptr = new RLTrader::SimExchange(filename, 250, start_read, max_read);
    }

    instr_ptr.reset(instr_raw_ptr);
    exchange_ptr.reset(exch_raw_ptr);
    strategy_ptr = std::make_unique<RLTrader::Strategy>(*instr_ptr, *exchange_ptr, balance, 20);
    adaptor_ptr = std::make_unique<RLTrader::EnvAdaptor>(*strategy_ptr, *exchange_ptr, spec.config["depth"_]);
  }

  void Reset() override {
    steps = 0;
    previous_rpnl = 0;
    previous_upnl = 0;
    previous_fees = 0;
    adaptor_ptr->reset();
    isDone = false;
    WriteState();
  }

  static u_int select_action(const std::vector<double>& logits) {
      auto maxLogitIt = std::max_element(logits.begin(), logits.end());
      return std::distance(logits.begin(), maxLogitIt);
  }


  void Step(const Action& action) override {
      std::vector<double> buyActionLogits { action["action"_][0],
                                            action["action"_][1],
                                            action["action"_][2],
                                            action["action"_][3],
                                          };

      std::vector<double> sellActionLogits { action["action"_][4],
                                             action["action"_][5],
                                             action["action"_][6],
                                             action["action"_][7],
                                           };

      auto buy_action = select_action(buyActionLogits);
      auto sell_action = select_action(sellActionLogits);
      auto buy_spread = spreads[buy_action];
      auto sell_spread = spreads[sell_action];
      adaptor_ptr->quote(buy_spread, sell_spread, 10, 10);
      auto info = adaptor_ptr->getInfo();
      isDone = !adaptor_ptr->next();
      ++steps;
      WriteState();
  }

  void WriteState() {
    auto data = adaptor_ptr->getState();
    State state = Allocate(1);

    if (!isDone) {
      assert(data.size() == 490);
    }

    auto info = adaptor_ptr->getInfo();
    state["info:mid_price"_] = static_cast<float>(info["mid_price"]);
    state["info:balance"_] = static_cast<float>(info["balance"]);
    state["info:unrealized_pnl"_] = static_cast<float>(info["unrealized_pnl"]);
    state["info:realized_pnl"_] = static_cast<float>(info["realized_pnl"]);
    state["info:leverage"_] = static_cast<float>(info["leverage"]);
    state["info:trade_count"_] = static_cast<float>(info["trade_count"]);
    state["info:drawdown"_] = static_cast<float>(info["drawdown"]);
    state["info:fees"_] = static_cast<float>(info["fees"]);
    state["info:buy_amount"_] = static_cast<float>(info["buy_amount"]);
    state["info:sell_amount"_] = static_cast<float>(info["sell_amount"]);

    auto pnl = info["realized_pnl"] - previous_rpnl; 
    auto upnl = info["unrealized_pnl"]- previous_upnl;
    state["reward"_] = (previous_fees - info["fees"]) + pnl + upnl; 

    if (isDone) return;
    previous_rpnl = info["realized_pnl"];
    previous_upnl = info["unrealized_pnl"];
    previous_fees = info["fees"];
       
    for(int ii=0; ii < data.size(); ++ii) {
      state["obs"_](ii) = static_cast<float>(data[ii]);
    }
  }

  bool IsDone() override { return isDone; }
};

using RlTraderLitePool = AsyncLitePool<RlTraderEnv>;

}  // namespace rltrader

#endif  // LITEPOOL_RLTRADER_RLTRADER_LITEPOOL_H_
