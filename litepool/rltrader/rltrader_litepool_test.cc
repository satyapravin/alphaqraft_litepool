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

#include "litepool/rltrader/rltrader_litepool.h"

#include <gtest/gtest.h>

#include <random>
#include <vector>

using RlTraderAction = rltrader::RlTraderEnv::Action;
using RlTraderState = rltrader::RlTraderEnv::State;

TEST(RlTraderLitePoolTest, SplitZeroAction) {
  auto config = rltrader::RlTraderEnvSpec::kDefaultConfig;
  int num_envs = 4;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = 4;
  config["num_threads"_] = 1;
  config["seed"_] = 42;

  rltrader::RlTraderEnvSpec spec(config);
  rltrader::RlTraderLitePool litepool(spec);

  TArray all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }

  litepool.Reset(all_env_ids);
  auto state_vec = litepool.Recv();

  std::vector<Array> raw_action;
  raw_action.reserve(3);

  // Create arrays with proper dimensions
  raw_action.push_back(Array(Spec<int>({num_envs})));         // env_id
  raw_action.push_back(Array(Spec<int>({num_envs, 1})));      // players.env_id
  raw_action.push_back(Array(Spec<double>({num_envs, 10})));   // action

  // Initialize players.env_id array
  for (int i = 0; i < num_envs; ++i) {
    raw_action[1](i, 0) = i;
  }

  RlTraderAction action(raw_action);

  for (int i = 0; i < num_envs; ++i) {
    // Set env_id
    action["env_id"_][i] = i;
    
    // Set action values using operator()
    action["action"_](i, 0) = 5.0f;
    action["action"_](i, 1) = 5.0f;
    action["action"_](i, 2) = 80.0f;
    action["action"_](i, 3) = 80.0f;
    action["action"_](i, 4) = 2.0f;
    action["action"_](i, 5) = 3.0f;
    action["action"_](i, 6) = 80.0f;
    action["action"_](i, 7) = 80.0f;
    action["action"_](i, 8) = 2.0f;
    action["action"_](i, 9) = 3.0f;
  }

  litepool.Send(std::move(action));
  RlTraderState state(litepool.Recv());

  for (int i = 0; i < num_envs; ++i) {
    EXPECT_EQ(static_cast<int>(state["info:env_id"_][i]), i);
    auto obs = state["obs"_](i);
    EXPECT_EQ(obs.size, 242*10);
  }
}

void Runner(int num_envs, int batch, int seed, int total_iter, int num_threads) {
  auto config = rltrader::RlTraderEnvSpec::kDefaultConfig;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = batch;
  config["num_threads"_] = num_threads;
  config["seed"_] = seed;
  config["max_num_players"_] = 1;

  rltrader::RlTraderEnvSpec spec(config);
  rltrader::RlTraderLitePool litepool(spec);

  TArray all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  litepool.Reset(all_env_ids);

  std::vector<Array> raw_action;
  raw_action.reserve(3);

  auto start = std::chrono::system_clock::now();
  for (int ii = 0; ii < total_iter; ++ii) {
    RlTraderState state(litepool.Recv());

    raw_action.clear();
    raw_action.push_back(Array(Spec<int>({num_envs})));         // env_id
    raw_action.push_back(Array(Spec<int>({num_envs, 1})));      // players.env_id
    raw_action.push_back(Array(Spec<double>({num_envs, 10})));   // action

    // Initialize players.env_id array
    for (int i = 0; i < num_envs; ++i) {
      raw_action[1](i, 0) = i;
    }

    RlTraderAction action(raw_action);

    for (int i = 0; i < num_envs; ++i) {
      action["env_id"_][i] = i;
      
      action["action"_](i, 0) = 5.0f;
      action["action"_](i, 1) = 5.0f;
      action["action"_](i, 2) = 80.0f;
      action["action"_](i, 3) = 80.0f;
      action["action"_](i, 4) = 2.0f;
      action["action"_](i, 5) = 3.0f;
      action["action"_](i, 6) = 80.0f;
      action["action"_](i, 7) = 80.0f;
      action["action"_](i, 8) = 2.0f;
      action["action"_](i, 9) = 3.0f;
    }

    litepool.Send(std::move(action));
  }

  std::chrono::duration<double> dur = std::chrono::system_clock::now() - start;
  double t = dur.count();
  double fps = (total_iter * batch) / t;
  std::cout << "time(s): " << t << ", FPS: " << fps << std::endl;
}

TEST(RlTraderLitePoolTest, SinglePlayer) {
  Runner(1, 1, 20, 100, 1);
  Runner(3, 3, 22, 1000, 3);
  Runner(12, 12, 21, 1000, 122);
}
