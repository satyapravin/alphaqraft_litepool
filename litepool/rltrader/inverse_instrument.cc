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

#include <cmath>
#include "inverse_instrument.h"

using namespace std;
using namespace RLTrader;

InverseInstrument::InverseInstrument(const std::string& symbol, const double& tickSize, 
    const double& minAmount, const double& makerFee, const double& takerFee)
    : BaseInstrument(symbol, tickSize, minAmount, makerFee, takerFee) {}

double InverseInstrument::getPositionFromAmount(const double& amount, const double& price) {
    return amount;
}

double InverseInstrument::getLeverage(const double& amount, const double& equity, const double& price) {
    // amount is in USD contracts, equity is in BTC
    // Convert to same units: notional_BTC / equity_BTC
    return amount / (equity * price);
}

double InverseInstrument::getTradeAmount(const double &amount, const double &refPrice) {
    return std::round(amount / minAmount) * minAmount;
}

double InverseInstrument::pnl(const double& qty, const double& entryPrice, const double& exitPrice) const {
    // Inverse PnL formula: qty * (1/entry - 1/exit) in BTC
    // = qty * (exit - entry) / (entry * exit)
    return entryPrice < tickSize ? 0.0 : qty * (exitPrice - entryPrice) / (entryPrice * exitPrice);
}

double InverseInstrument::equity(const double& mid, const double& balance, const double& position,
    const double& avgPrice, const double& fee) const {
    return balance + this->pnl(position, avgPrice, mid) - fee;
}

double InverseInstrument::fees(const double& qty, const double& price, bool isMaker) const {
    // Fee in BTC = (contract_value_in_BTC) * fee_rate = (qty / price) * fee_rate
    if (isMaker)
    {
        return abs(qty) * this->makerFee / price;
    }
    else {
        return abs(qty) * this->takerFee / price;
    }
}
