#pragma once

#include <memory>
#include "exchange.h"
#include "position.h"

namespace Simulator {
	class Strategy {
	public:
		Strategy(BaseInstrument& instr, Exchange& exch, const double& balance,
			const double& pos_amount, const double& avg_price,
			const double& areaFactor, int maxTicks);
			
		void reset(int time_index, const double& position_amount, const double& avg_price);

		void quote(int bid_tick_spread, int ask_tick_spread,
			       const double& buyVolumeAngle, const double& sellVolumeAngle);

		void fetchInfo(PositionInfo& info, const double& bidPrice, const double& askPrice);
                
		Position& getPosition() { return position; }
	private:
		BaseInstrument& instrument;
		Exchange& exchange;
		Position position;
		int order_id;
		double maxFactor;
		int max_ticks;
		void sendGrid(const double& angle, const DataRow& obs, OrderSide side);
	};
}
