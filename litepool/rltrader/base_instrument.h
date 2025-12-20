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
    class BaseInstrument {
    public:
        BaseInstrument(const std::string& aSymbol, const double& aTickSize, // NOLINT(*-pass-by-value)
            const double& aMinAmount, const double& aMakerFee, const double& aTakerFee)
            :symbol(aSymbol), tickSize(aTickSize), minAmount(aMinAmount), 
            makerFee(aMakerFee), takerFee(aTakerFee)
        {
        }

        virtual ~BaseInstrument() = default;
        [[nodiscard]] std::string getName() const { return symbol; }
        [[nodiscard]] double getTickSize() const { return tickSize; }
        [[nodiscard]] double getMinAmount() const { return minAmount; }
        [[nodiscard]] virtual double getPositionFromAmount(const double& amount, const double& price) = 0;
        [[nodiscard]] virtual double getLeverage(const double& amount, const double& equity, const double& price) = 0;
        [[nodiscard]] virtual double getTradeAmount(const double& amount, const double& refPrice) = 0;
        [[nodiscard]] virtual double equity(const double& mid, const double& balance,
                              const double& position, const double& avgPrice, 
                              const double& fee) const = 0;
        [[nodiscard]] virtual double pnl(const double& qty, const double& entryPrice,
                           const double& exitPrice) const = 0;
        [[nodiscard]] virtual double fees(const double& qty, const double& price, bool isMaker) const = 0;

    protected:
        std::string symbol;
        double tickSize;
        double minAmount;
        double makerFee;
        double takerFee;
    };
}
