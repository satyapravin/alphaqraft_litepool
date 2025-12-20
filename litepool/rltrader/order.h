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

#pragma once

#include <string>

namespace RLTrader {
    enum class OrderState {
        NEW = 1,
        NEW_ACK = 2,
        AMEND = 3,
        AMEND_ACK = 4,
        CANCELLED = 5,
        CANCELLED_ACK = 6,
        FILLED = 7
    };

    enum OrderSide {
        BUY = 1,
        SELL = 2
    };

    struct Order
    {
        bool is_taker;
        std::string orderId;
        OrderSide side;
        double price;
        double amount;
        OrderState state;
        long long microSecond;
    };
}