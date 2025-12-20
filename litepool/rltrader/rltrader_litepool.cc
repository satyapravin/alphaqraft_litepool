// Copyright 2024 Alphaqraft
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

#include "litepool/core/py_litepool.h"

/**
 * Wrap the `RlTraderEnvSpec` and `RlTraderLitePool` with the corresponding `PyEnvSpec`
 * and `PyLitePool` template.
 */
using RlTraderEnvSpec = PyEnvSpec<rltrader::RlTraderEnvSpec>;
using RlTraderLitePool = PyLitePool<rltrader::RlTraderLitePool>;

/**
 * Finally, call the REGISTER macro to expose them to python
 */
PYBIND11_MODULE(rltrader_litepool, m) { REGISTER(m, RlTraderEnvSpec, RlTraderLitePool) }
