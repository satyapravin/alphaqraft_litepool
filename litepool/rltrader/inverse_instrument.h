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

#include "base_instrument.h"

namespace RLTrader {
    class InverseInstrument : public BaseInstrument {
    public:
        InverseInstrument(const std::string& symbol, const double& tickSize, 
            const double& minAmount, const double& makerFee, const double& takerFee);
        [[nodiscard]] double getPositionFromAmount(const double& amount, const double& price) override;
        [[nodiscard]] double getLeverage(const double& amount, const double& equity, const double& price) override;
        [[nodiscard]] double getTradeAmount(const double &amount, const double &refPrice) override;
        [[nodiscard]] double pnl(const double& qty, const double& entryPrice, const double& exitPrice) const override;
        [[nodiscard]] double equity(const double& mid, const double& balance, const double& position,
                      const double& avgPrice, const double& fee) const override;
        [[nodiscard]] double fees(const double& qty, const double& price, bool isMaker) const override;
    };
}