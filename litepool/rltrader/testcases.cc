#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <algorithm>

#include <string>
#include <random>
#include <chrono>
#include "inverse_instrument.h"
#include "csv_reader.h"
#include "position.h"
#include "sim_exchange.h"
#include "strategy.h"
#include "orderbook.h"
#include "market_signal_builder.h"
#include "circ_buffer.h"
#include "env_adaptor.h"


#include "normal_instrument.h"

using namespace RLTrader;
using namespace doctest;

bool double_equals(double a, double b, double epsilon = 1e-9) {
	return std::abs(a - b) <= epsilon;
}

TEST_CASE("Testing TemporalTable") {
    constexpr u_int rows = 3;
    constexpr u_int cols = 20;
    TemporalTable table(rows);

    SUBCASE("Initial state") {
        for (u_int i = 0; i < rows; ++i) {
            const auto& row = table.get(i);
            CHECK(row.size() == cols);
            CHECK(std::all_of(row.begin(), row.end(), [](double val) { return val == 0.0; }));
        }
    }

    SUBCASE("Adding and retrieving rows") {
    	FixedVector<double, 20> row1, row2, row3;

    	for (int ii=0; ii < 20; ++ii) {
    		row1[ii] = 1.0 + 0.4 * ii;
    		row2[ii] = 2.0 + 0.4 * ii;
    		row3[ii] = 3.0 + 0.4 * ii;
    	}

        table.addRow(row1);
		auto result = table.get(0);
        CHECK(std::equal(table.get(0).begin(), table.get(0).end(), row1.begin()));

        table.addRow(row2);
        CHECK(std::equal(table.get(0).begin(), table.get(0).end(), row2.begin()));
        CHECK(std::equal(table.get(1).begin(), table.get(1).end(), row1.begin()));

        table.addRow(row3);
        CHECK(std::equal(table.get(0).begin(), table.get(0).end(), row3.begin()));
        CHECK(std::equal(table.get(1).begin(), table.get(1).end(), row2.begin()));
        CHECK(std::equal(table.get(2).begin(), table.get(2).end(), row1.begin()));
    }

    SUBCASE("Overwriting old rows") {
    	FixedVector<double, 20> row1, row2, row3, row4;

    	for (int ii=0; ii < 20; ++ii) {
    		row1[ii] = 1.0 + 0.4 * ii;
    		row2[ii] = 2.0 + 0.4 * ii;
    		row3[ii] = 3.0 + 0.4 * ii;
    		row4[ii] = 4.0 * 0.4 * ii;
    	}

        table.addRow(row1);
        table.addRow(row2);
        table.addRow(row3);
        table.addRow(row4);

        CHECK(std::equal(table.get(0).begin(), table.get(0).end(), row4.begin()));
        CHECK(std::equal(table.get(1).begin(), table.get(1).end(), row3.begin()));
        CHECK(std::equal(table.get(2).begin(), table.get(2).end(), row2.begin()));
    }
}

struct TestData {
    int value;
    bool operator==(const TestData& other) const { return value == other.value; }
};

TEST_CASE("Testing TemporalBuffer with custom class TestData") {
    RLTrader::TemporalBuffer<TestData> buffer(2); // Buffer for 2 lags

    SUBCASE("Initial state") {
        CHECK_NOTHROW(buffer.get(0));
        CHECK_NOTHROW(buffer.get(1));
        CHECK_NOTHROW(buffer.get(2));
    }

    SUBCASE("Adding and retrieving custom objects") {
        buffer.add(TestData{1});
        CHECK(buffer.get(0) == TestData{1});

        buffer.add(TestData{2});
        CHECK(buffer.get(0) == TestData{2});
        CHECK(buffer.get(1) == TestData{1});

        buffer.add(TestData{3});
        CHECK(buffer.get(0) == TestData{3});
        CHECK(buffer.get(1) == TestData{2});
        CHECK(buffer.get(2) == TestData{1});
    }

    SUBCASE("Overwriting old values with custom objects") {
        buffer.add(TestData{4});
        buffer.add(TestData{5});
        buffer.add(TestData{6});
        CHECK(buffer.get(0) == TestData{6});
        CHECK(buffer.get(1) == TestData{5});
        CHECK(buffer.get(2) == TestData{4});
    }

    SUBCASE("Out of range access with custom objects") {
        CHECK_THROWS_AS(buffer.get(3), std::out_of_range);
        CHECK_THROWS_AS(buffer.get(-1), std::out_of_range);
    }

    SUBCASE("Adding multiple custom objects in a loop") {
        for (int i = 1; i <= 1000; ++i) {
            buffer.add(TestData{i});
            CHECK(buffer.get(0) == TestData{i});
            if (i > 1) {
                CHECK(buffer.get(1) == TestData{i - 1});
            }
            if (i > 2) {
                CHECK(buffer.get(2) == TestData{i - 2});
            }
        }
    }
}

TEST_CASE("env adaptor test") {
	SimExchange exch("data.csv", 5, 0, 100);
	InverseInstrument instr("BTC", 0.5, 10.0, 0, 0.0005);
	Strategy strategy(instr, exch, 1, 5);
	EnvAdaptor adaptor = EnvAdaptor(strategy, exch);
	adaptor.reset();

	int counter = 0;
	std::array<double, 490> state;
	adaptor.getState(state);
	CHECK(state.size() == 98*5);
	adaptor.next();
	adaptor.getState(state);
	CHECK(state.size() == 98*5);
	adaptor.getState(state);
	CHECK(state.size() == 98*5);
	adaptor.next();
	adaptor.getState(state);
	CHECK(state.size() == 98*5);
	adaptor.quote(1, 1, 10, 10);

	for (int ii=0; ii < 500; ++ii) {
		adaptor.next();
		adaptor.getState(state);
		adaptor.quote(0, 0, 10, 10);
	}

	adaptor.next();
	std::array<double, 490> signals;
	adaptor.getState(signals);
	CHECK(std::all_of(signals.begin(), signals.end(), [](double val) {return std::isfinite(val);}));
	CHECK(std::all_of(signals.begin(), signals.end(), [](double val) { return std::abs(val) < 10;}));
	CHECK(std::all_of(signals.begin(), signals.end(), [](double val) { return std::abs(val) >= 0;}));
}

TEST_CASE("test of OrderBook and signals") {
	double bid_price = 1000;
	double ask_price = 1000;
	OrderBook lob;
	std::mt19937 rng;
	std::random_device rd;
	rng.seed(rd());
	std::uniform_int_distribution<int> dist(1000, 50000);

	for(int ii=0; ii < 20; ++ii) {
		bid_price -= 0.5;
		ask_price += 0.5;
		lob.bid_prices[ii] = bid_price;
		lob.ask_prices[ii] = ask_price;
		lob.bid_sizes[ii] = dist(rng);
		lob.ask_sizes[ii] = dist(rng);
	}
	auto& book = lob;
	CHECK(book.ask_prices.size() == 20);
	CHECK(book.bid_prices.size() == 20);
	CHECK(book.ask_sizes.size() == 20);
	CHECK(book.bid_sizes.size() == 20);
	MarketSignalBuilder builder;
	std::vector<std::chrono::duration<double>> durations;

	int ii = 0;
	for (ii=0; ii < 15000; ++ii) {
		double mid_price = 0.5 * (bid_price + ask_price) + dist(rng) / 2000.0;
		bid_price = mid_price;
		ask_price = mid_price;

		for(int jj=0; jj < 20; ++jj) {
			bid_price -= 0.5;
			ask_price += 0.5;
			lob.bid_prices[jj] = bid_price;
			lob.ask_prices[jj] = ask_price;
			lob.bid_sizes[jj] = dist(rng);
			lob.ask_sizes[jj] = dist(rng);
		}

		auto signals = builder.add_book(book);

		if (ii > 30) {
			CHECK(std::all_of(signals.begin(), signals.end(), [](const double& val) {return std::isfinite(val);}));
			CHECK(std::count_if(signals.begin(), signals.end(), [](const double& val) { return std::abs(val) == 0.0;}) <= 4);
			CHECK(std::all_of(signals.begin(), signals.end(), [](const double& val) { return std::abs(val) < 100;}));
		}
	}

	CHECK(ii == 15000);
}

TEST_CASE("testing the inverse_instrument") {
	InverseInstrument instr("BTC", 0.5, 10.0, 0.0, 0.0005);
	CHECK(instr.getTickSize() == Approx(0.5));
	CHECK(instr.getName() == "BTC");
	CHECK(instr.getMinAmount() == Approx(10.0));
	CHECK(instr.pnl(1000, 10000.0, 20000.0) == Approx(0.05));
	CHECK(instr.fees(1000, 10000.0, true) == Approx(0.0));
	CHECK(instr.fees(1000, 10000.0, false) == Approx(0.00005));
}

TEST_CASE("testing the normal_instrument") {
	NormalInstrument instr("BTCUSDT", 0.1, 0.0001, -0.0001, 0.0075);
	CHECK(instr.getTickSize() == Approx(0.1));
	CHECK(instr.getName() == "BTCUSDT");
	CHECK(instr.getMinAmount() == Approx(0.0001));
	CHECK(instr.pnl(0.1, 10000, 20000) == Approx(1000));
	CHECK(instr.fees(0.1, 10000.0, true) == Approx(1000 * -0.0001));
	CHECK(instr.fees(0.1, 10000.0, false) == Approx(1000 * 0.0075));
}

TEST_CASE("testing the csv reader") {
	CsvReader reader("data.csv", 0, 1000);
	reader.reset();
	int counter = 0;
	while(reader.hasNext()) {
		auto& next = reader.next();
		auto _ = next.getBestAskPrice();
		_ = next.getBestBidPrice();
		counter++;
	}

	SimExchange exch("data.csv", 300, 0, 100);
	exch.reset();
	counter = 0;
	OrderBook book;
	size_t read_slot;
	while(exch.next_read(read_slot, book))
		++counter;
	exch.reset();
	while(exch.next_read(read_slot, book))
		++counter;
}

TEST_CASE("testing the normal position") {
	NormalInstrument instr("BTCUSDT", 0.1, 0.0001, -0.0001, 0.00075);
	Position pos(instr, 2000, 0, 0);

	SUBCASE("initial position") {
		CHECK(pos.getInitialBalance() == Approx(2000));
		PositionInfo info = pos.getPositionInfo(1000, 1001);
		TradeInfo& tradeInfo = pos.getTradeInfo();
		CHECK(info.averagePrice == Approx(0.0));
		CHECK(info.balance == Approx(2000));
		CHECK(info.inventoryPnL == Approx(0.0));
		CHECK(info.leverage == Approx(0.0));
		CHECK(info.tradingPnL == Approx(0.0));
		CHECK(tradeInfo.buy_trades == Approx(0.0));
		CHECK(tradeInfo.sell_trades == Approx(0.0));
		CHECK(tradeInfo.buy_amount == Approx(0.0));
		CHECK(tradeInfo.sell_amount == Approx(0.0));
		CHECK(tradeInfo.average_buy_price == Approx(0.0));
		CHECK(tradeInfo.average_sell_price == Approx(0.0));
	}

	SUBCASE("first buy order") {
		Order order;
		order.amount = 0.001;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1000.0;
		order.side = OrderSide::BUY;
		order.state = OrderState::FILLED;
		order.is_taker = false;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(2000));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(1000.0));
		CHECK(info.balance == Approx(2000));
		CHECK(info.inventoryPnL == Approx(0.015));
		CHECK(info.leverage == Approx(0.000507496));
		CHECK(info.tradingPnL == Approx(0.0));
		TradeInfo& tradeInfo = pos.getTradeInfo();
		CHECK(tradeInfo.buy_trades == 1);
		CHECK(tradeInfo.sell_trades == 0);
		CHECK(tradeInfo.buy_amount == Approx(0.001));
		CHECK(tradeInfo.sell_amount == Approx(0.0));
		CHECK(tradeInfo.average_buy_price == Approx(1000.0));
		CHECK(tradeInfo.average_sell_price == Approx(0.0));
	}

	SUBCASE("Three buys and a smaller sell order") {
		for (int ii = 1; ii <= 3; ++ii) {
			Order order;
			order.amount = 0.1;
			order.microSecond = 1;
			order.orderId = "1";
			order.price = 1000.0;
			order.side = OrderSide::BUY;
			order.state = OrderState::FILLED;
			order.is_taker = true;
			pos.onFill(order);
			CHECK(pos.getInitialBalance() == Approx(2000));
			PositionInfo info = pos.getPositionInfo(1010, 1020);
			CHECK(info.averagePrice == Approx(1000.0));
			CHECK(info.balance == Approx(2000));
			CHECK(info.inventoryPnL == Approx(1.5 * ii));
			CHECK(info.tradingPnL == Approx(0.0));
			TradeInfo& tradeInfo = pos.getTradeInfo();
			CHECK(tradeInfo.buy_trades == ii);
			CHECK(tradeInfo.sell_trades == 0);
			CHECK(tradeInfo.buy_amount == Approx(0.1 * ii));
			CHECK(tradeInfo.sell_amount == Approx(0.0));
			CHECK(tradeInfo.average_buy_price == Approx(1000.0));
			CHECK(tradeInfo.average_sell_price == Approx(0.0));
		}

		Order order;
		order.amount = 0.2;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1015.0;
		order.side = OrderSide::SELL;
		order.state = OrderState::FILLED;
		order.is_taker = false;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(2000));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(1000.0));
		CHECK(info.balance == Approx(2003));
		CHECK(info.inventoryPnL == Approx(1.5));
		CHECK(info.leverage == Approx(0.0506361));
		CHECK(info.tradingPnL == Approx(3));
		TradeInfo& tradeInfo = pos.getTradeInfo();
		CHECK(tradeInfo.sell_trades == 1);
		CHECK(tradeInfo.buy_trades == 3);
		CHECK(tradeInfo.buy_amount == Approx(0.3));
		CHECK(tradeInfo.sell_amount == Approx(0.2));
		CHECK(tradeInfo.average_buy_price == Approx(1000.0));
		CHECK(tradeInfo.average_sell_price == Approx(1015));
	}
}

TEST_CASE("testing the inverse position") {
	InverseInstrument instr("BTC", 0.5, 10.0, 0.0, 0.0005);
	Position pos(instr, 0.1, 0, 0.0);

	SUBCASE("initial position") {
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info = pos.getPositionInfo(1000, 1001);
		TradeInfo& tradeInfo = pos.getTradeInfo();
		CHECK(info.averagePrice == Approx(0.0));
		CHECK(info.balance == Approx(0.1));
		CHECK(info.inventoryPnL == Approx(0.0));
		CHECK(info.leverage == Approx(0.0));
		CHECK(info.tradingPnL == Approx(0.0));
		CHECK(tradeInfo.buy_trades == Approx(0.0));
		CHECK(tradeInfo.sell_trades == Approx(0.0));
		CHECK(tradeInfo.buy_amount == Approx(0.0));
		CHECK(tradeInfo.sell_amount == Approx(0.0));
		CHECK(tradeInfo.average_buy_price == Approx(0.0));
		CHECK(tradeInfo.average_sell_price == Approx(0.0));
	}

	SUBCASE("first buy order") {
		Order order;
		order.amount = 10.0;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1000.0;
		order.side = OrderSide::BUY;
		order.state = OrderState::FILLED;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(1000.0));
		CHECK(info.balance == Approx(0.1));
		CHECK(info.inventoryPnL == Approx(0.000147783));
		CHECK(info.leverage == Approx(0.09837678));
		CHECK(info.tradingPnL == Approx(0.0));
		TradeInfo& tradeInfo = pos.getTradeInfo();
		CHECK(tradeInfo.buy_trades == 1);
        CHECK(tradeInfo.sell_trades == 0);
        CHECK(tradeInfo.buy_amount == Approx(10.0));
        CHECK(tradeInfo.sell_amount == Approx(0.0));
        CHECK(tradeInfo.average_buy_price == Approx(1000.0));
        CHECK(tradeInfo.average_sell_price == Approx(0.0));
	}

	SUBCASE("Three buys and a smaller sell order") {
		for (int ii = 1; ii <= 3; ++ii) {
			Order order;
			order.amount = 10.0;
			order.microSecond = 1;
			order.orderId = "1";
			order.price = 1000.0;
			order.side = OrderSide::BUY;
			order.state = OrderState::FILLED;
			pos.onFill(order);
			CHECK(pos.getInitialBalance() == Approx(0.1));
			PositionInfo info = pos.getPositionInfo(1010, 1020);
			CHECK(info.averagePrice == Approx(1000.0));
			CHECK(info.balance == Approx(0.1));
			CHECK(info.inventoryPnL == Approx(0.000147783 * ii));
			CHECK(info.tradingPnL == Approx(0.0));
			TradeInfo& tradeInfo = pos.getTradeInfo();
			CHECK(tradeInfo.buy_trades == ii);
			CHECK(tradeInfo.sell_trades == 0);
			CHECK(tradeInfo.buy_amount == Approx(10.0 * ii));
			CHECK(tradeInfo.sell_amount == Approx(0.0));
			CHECK(tradeInfo.average_buy_price == Approx(1000.0));
			CHECK(tradeInfo.average_sell_price == Approx(0.0));
		}

		Order order;
		order.amount = 15.0;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1015.0;
		order.side = OrderSide::SELL;
		order.state = OrderState::FILLED;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(1000.0));
		CHECK(info.balance == Approx(0.10022167));
		CHECK(info.inventoryPnL == Approx(0.000147783 * 1.5));
		CHECK(info.leverage == Approx(0.14713094));
		CHECK(info.tradingPnL == Approx(0.0002216487));
		TradeInfo& tradeInfo = pos.getTradeInfo();
		CHECK(tradeInfo.sell_trades == 1);
		CHECK(tradeInfo.buy_trades == 3);
		CHECK(tradeInfo.buy_amount == Approx(30.0));
		CHECK(tradeInfo.sell_amount == Approx(15.0));
		CHECK(tradeInfo.average_buy_price == Approx(1000.0));
		CHECK(tradeInfo.average_sell_price == Approx(1015));
	}

	SUBCASE("Three sells and a smaller buy order") {
		for (int ii = 1; ii <= 3; ++ii) {
			Order order;
			order.amount = 10.0;
			order.microSecond = 1;
			order.orderId = "1";
			order.price = 1000.0;
			order.side = OrderSide::SELL;
			order.state = OrderState::FILLED;
			pos.onFill(order);
			CHECK(pos.getInitialBalance() == Approx(0.1));
			PositionInfo info = pos.getPositionInfo(1010, 1020);
			CHECK(info.averagePrice == Approx(1000.0));
			CHECK(info.balance == Approx(0.1));
			CHECK(info.inventoryPnL == Approx(-0.000147783 * ii));
			CHECK(info.tradingPnL == Approx(0.0));
			TradeInfo& tradeInfo = pos.getTradeInfo();
			CHECK(tradeInfo.sell_trades == ii);
			CHECK(tradeInfo.buy_trades == 0);
			CHECK(tradeInfo.buy_amount == Approx(0.0));
			CHECK(tradeInfo.sell_amount == Approx(10.0*ii));
			CHECK(tradeInfo.average_buy_price == Approx(0.0));
			CHECK(tradeInfo.average_sell_price == Approx(1000));
		}

		Order order;
		order.amount = 15.0;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1015.0;
		order.side = OrderSide::BUY;
		order.state = OrderState::FILLED;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(1000.0));
		CHECK(info.balance == Approx(0.099778325));
		CHECK(info.inventoryPnL == Approx(-0.000147783 * 1.5));
		CHECK(info.leverage == Approx(-0.1484413656));
		CHECK(info.tradingPnL == Approx(-0.00022167487));
	}

	SUBCASE("Equal buy and sell order") {
		{
			Order order;
			order.amount = 10.0;
			order.microSecond = 1;
			order.orderId = "1";
			order.price = 1000.0;
			order.side = OrderSide::BUY;
			order.state = OrderState::FILLED;
			pos.onFill(order);
			CHECK(pos.getInitialBalance() == Approx(0.1));
			PositionInfo info = pos.getPositionInfo(1010, 1020);
			CHECK(info.averagePrice == Approx(1000.0));
			CHECK(info.balance == Approx(0.1));
			CHECK(info.inventoryPnL == Approx(0.000147783));
			CHECK(info.leverage == Approx(0.09837678));
			CHECK(info.tradingPnL == Approx(0.0));
		}

		Order order;
		order.amount = 10.0;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1015.0;
		order.side = OrderSide::SELL;
		order.state = OrderState::FILLED;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info= pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(0));
		CHECK(info.balance == Approx(0.10014778325));
		CHECK(info.inventoryPnL == Approx(0));
		CHECK(info.leverage == Approx(0));
		CHECK(info.tradingPnL == Approx(0.00014778325));
	}

	SUBCASE("Equal sell and buy order") {
		{
			Order order;
			order.amount = 10.0;
			order.microSecond = 1;
			order.orderId = "1";
			order.price = 1015.0;
			order.side = OrderSide::SELL;
			order.state = OrderState::FILLED;
			pos.onFill(order);
			CHECK(pos.getInitialBalance() == Approx(0.1));
			PositionInfo info = pos.getPositionInfo(1010, 1020);
			CHECK(info.averagePrice == Approx(1015.0));
			CHECK(info.balance == Approx(0.1));
			CHECK(info.inventoryPnL == Approx(0));
			CHECK(info.leverage == Approx(-0.09852216748768472));
			CHECK(info.tradingPnL == Approx(0.0));
		}

		Order order;
		order.amount = 10.0;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1000.0;
		order.side = OrderSide::BUY;
		order.state = OrderState::FILLED;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(0.0));
		CHECK(info.balance == Approx(0.10014778325));
		CHECK(info.inventoryPnL == Approx(0));
		CHECK(info.leverage == Approx(0));
		CHECK(info.tradingPnL == Approx(0.00014778325));
	}

	SUBCASE("Buy more than initial sell order") {
		{
			Order order;
			order.amount = 10.0;
			order.microSecond = 1;
			order.orderId = "1";
			order.price = 1015.0;
			order.side = OrderSide::SELL;
			order.state = OrderState::FILLED;
			pos.onFill(order);
			CHECK(pos.getInitialBalance() == Approx(0.1));
			PositionInfo info = pos.getPositionInfo(1010, 1020);
			CHECK(info.averagePrice == Approx(1015.0));
			CHECK(info.balance == Approx(0.1));
			CHECK(info.inventoryPnL == Approx(0));
			CHECK(info.leverage == Approx(-0.09852216748768472));
			CHECK(info.tradingPnL == Approx(0.0));
		}

		Order order;
		order.amount = 20.0;
		order.microSecond = 1;
		order.orderId = "1";
		order.price = 1000.0;
		order.side = OrderSide::BUY;
		order.state = OrderState::FILLED;
		pos.onFill(order);
		CHECK(pos.getInitialBalance() == Approx(0.1));
		PositionInfo info = pos.getPositionInfo(1010, 1020);
		CHECK(info.averagePrice == Approx(1000.0));
		CHECK(info.balance == Approx(0.10014778325));
		CHECK(info.inventoryPnL == Approx(0.0001477832));
		CHECK(info.leverage == Approx(0.09823182));
		CHECK(info.tradingPnL == Approx(0.00014778325));
	}
}

TEST_CASE("testing exchange") {
	SimExchange exch("data.csv", 5, 0, 100); // 10 microsecond delay is not practical in reality
	exch.reset();
	OrderBook row;
	size_t read_slot;
	exch.next_read(read_slot, row);

	CHECK(row.bid_prices[0] == Approx(63100));
	CHECK(row.bid_prices[1] == Approx(63099.5));

	for (int ii = 0; ii < 8; ++ii)
		CHECK(exch.next_read(read_slot, row));

	OrderBook next;
	CHECK(exch.next_read(read_slot, next));
	exch.reset();
	exch.next_read(read_slot, next);
	CHECK(next.bid_prices[0] == Approx(63100));
	CHECK(next.bid_prices[1] == Approx(63099.5));
	exch.reset();
	exch.quote("1", OrderSide::SELL, 42302, 100);
	exch.quote("2", OrderSide::SELL, 42305, 500);
	exch.quote("3", OrderSide::BUY, 40000, 300);
	exch.quote("4", OrderSide::BUY, 39000, 200);
	std::vector<Order> unacks = exch.getUnackedOrders();
	CHECK(unacks.size() == 4);
	size_t slot;
	exch.next_read(slot, row);
	const auto& bids = exch.getBidOrders();
	CHECK(bids.size() == 2);
	const auto& asks = exch.getAskOrders();
	CHECK(asks.size() == 0);
	unacks = exch.getUnackedOrders();
	CHECK(unacks.size() == 0);
}

TEST_CASE("test of inverse strategy") {
	SimExchange exch("data.csv", 5, 0, 1000);
	OrderBook book;
	size_t slot;
	exch.next_read(slot, book);
	InverseInstrument instr("BTC", 0.5, 10.0, 0, 0.0005);
	Strategy strategy(instr, exch, 0.015, 5);
	strategy.quote(2, 2, 1, 1, book.bid_prices, book.ask_prices);
	exch.next_read(slot, book);
	const auto& bids = exch.getBidOrders();
	const auto& asks = exch.getAskOrders();
	CHECK(bids.size() == 1);
	CHECK(asks.size() == 1);
}

TEST_CASE("test of normal strategy") {
	SimExchange exch("data.csv", 5, 0, 1000);
	OrderBook book;
	size_t slot;
	exch.next_read(slot, book);
	NormalInstrument instr("BTCUSDT", 0.1, .0001, -0.0001, 0.0075);
	Strategy strategy(instr, exch, 2000,  5);
	strategy.quote(2, 2, 1, 1, book.bid_prices, book.ask_prices);
	exch.next_read(slot, book);
	const auto& bids = exch.getBidOrders();
	const auto& asks = exch.getAskOrders();
	CHECK(bids.size() == 1);
	CHECK(asks.size() == 1);
}
