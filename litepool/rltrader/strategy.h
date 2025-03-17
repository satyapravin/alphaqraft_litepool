#pragma once

#include <memory>

#include "base_exchange.h"
#include "orderbook.h"
#include "position.h"

namespace RLTrader {
	class Strategy {
	public:
		Strategy(BaseInstrument& instr, BaseExchange& exch, const double& balance, int maxTicks);
			
		void reset();

		void quote(const std::vector<double>& buy_spreads, 
			   const std::vector<double>& sell_spreads, 
			   const std::vector<double>& buy_volumes, 
			   const std::vector<double>& sell_volumes,
		           FixedVector<double, 20>& bid_prices, 
			   FixedVector<double, 20>& ask_prices);

		Position& getPosition() { return position; }

		void next();

	private:
		BaseInstrument& instrument;
		BaseExchange& exchange;
		Position position;
		int order_id;
		int max_ticks;
	};
}
