# Test Coverage Report

Last updated after adding comprehensive position/PnL tests and CTest integration.

## Current Test Status

```
[doctest] test cases:    32 |    32 passed | 0 failed | 0 skipped
[doctest] assertions: 40257 | 40257 passed | 0 failed |
[doctest] Status: SUCCESS!
```

All tests passing as of last run. Tests now integrated into CMake build system.

## Test Breakdown

### Core Infrastructure
- TemporalTable - circular buffer with lag access, handles empty state properly
- TemporalBuffer - generic version for custom types, edge cases covered
- CSV Reader - file parsing works, random start positions, max read limits enforced
- SimExchange - order book updates, order execution logic, fill handling
- OrderBook - basic structure validation

### Position & PnL
- Normal Instrument PnL - linear calculations, fee handling
- Inverse Instrument PnL - inverse formula tested, fees calculated correctly
- Position LIFO Spread Capture - this one was tricky, had to verify the LIFO stack behavior matches expected. Tests cover:
  - LIFO closing (most recent position closes first)
  - Spread capture calculation
  - netPosition = amount * mid_price

### Strategy
- Inverse Strategy - 5-level ladder quoting, price computation looks good
- Normal Strategy - same ladder logic, order properties validated
- Strategy with Invalid Prices - handles zero/negative gracefully

### Risk Management
- Leverage Limit Enforcement - stops quoting at leverage >= 1.0, sets flag correctly, cancels orders

### Order Execution
- POST_ONLY rejection works (aggressive orders get rejected)
- Passive orders accepted
- State transitions: NEW → NEW_ACK → FILLED

### Trade Feed
- Trade Reader - CSV parsing, timestamps sync correctly, trades accumulate as expected
- Trade Signal Builder - all 8 signals normalized to [-1, 1], volume imbalance works, temporal signals decay properly

### Environment
- EnvAdaptor - state computation, signal generation, info tracking
- EnvAdaptor with Trade Signals - integration test verifies all 26 observation signals in [-1, 1], trade signals at indices 17-24

### Signal Normalization
- Signal Normalization Bounds - ran across multiple steps, all 26 signals finite and bounded
- Market Signal Builder Edge Cases - tested with normal conditions and extreme spreads (70000 vs 63100), signals stay normalized

### Rewards
- Reward Calculation Components - realized_pnl, unrealized_pnl, spread_capture, fees all present. Tested with profitable round-trip, spread capture positive as expected.

### AMM
- AMM Simulator Edge Cases - initialization, large price jumps (63100 → 70000), zero price handling. All signals stay in bounds.

### Value Accuracy Tests
Added these to catch numerical issues before they hit production:

- Market Signal Values Accuracy - verified sigmoid normalization formula matches expected values, depth signals positive and normalized, volume_imbalance for symmetric books
- Position Values Accuracy (Normal) - netAmount, averagePrice, netPosition calculations exact, inventoryPnL with mid price, spread capture on LIFO closes
- Position Values Accuracy (Inverse) - netAmount in USD contracts, netPosition, inverse PnL formula: qty * (exit - entry) / (entry * exit), leverage calculation
- PnL Calculation Accuracy (Normal) - spread capture = (exit - entry) * amount, fee calculations with maker rebates, realized PnL = balance - initialBalance
- PnL Calculation Accuracy (Inverse) - inverse spread capture formula, fees = abs(qty) * fee_rate / price, realized PnL accounting
- Average Price Calculation Accuracy - weighted average: sum(amount * price) / sum(amount), inventoryPnL with calculated average
- Leverage Calculation Accuracy - normal: amount * price / equity, equity = balance + unrealized_pnl - fees, leverage with maker fee rebates

## Production-Critical Areas

### Position Tracking
- LIFO stack management (long_stack, short_stack) - this is critical for spread capture accuracy
- Spread capture on position close
- Net position calculation differs for normal vs inverse instruments
- Average price tracking
- Trade count and volume

### Risk Management
- Leverage limit detection at ±1.0
- Early termination when limit hit
- Order cancellation
- Flag propagation to reward

### Order Execution
- POST_ONLY rejection for crossing orders
- Passive order acceptance
- State transitions
- Fill price validation

### Signal Normalization
- All 26 observation signals in [-1, 1]
- Finite value checks
- Edge cases: extreme spreads, zero prices
- Temporal signal decay

### Reward Calculation
- Realized PnL delta (normalized by initial_balance)
- Spread capture delta
- Unrealized PnL delta
- Fee delta
- Leverage limit penalty
- Early termination reward

### Strategy Logic
- 5-level ladder quoting per side
- Spread = base * vol_mult * action_mult
- Inventory skew
- Volatility adjustment
- Minimum spread enforcement
- Invalid price handling

## Test Data

Test data lives in `test_data/` (separate from training/testing data):
- Book data: `test_data/data.csv` - 12 rows, 100ms intervals, 84 columns
- Trade data: `test_data/trades/1.csv` - 11 trades, 100ms intervals
- Format matches production exactly

## Running Tests

Tests are now integrated into the build system:

```bash
# Build everything (tests build automatically)
cd build && cmake .. && make

# Run all tests via CTest
cd build && ctest --output-on-failure

# Or use the custom target
cd build && make run_tests

# Run just the doctest suite
cd build && ctest -R doctest_testcases

# Run manually if needed
cd /home/pravin/dev/alphaqraft_litepool && ./build/bin/testcases

# Run specific test
./build/bin/testcases --test-case="test leverage limit enforcement"
```

Tests run automatically as part of the build process. The working directory is set correctly so test_data/ paths resolve.

## Notes

- Tests use strict assertions - they verify exact expected behavior, not just "doesn't crash"
- Edge cases covered: invalid inputs, extreme values, boundary conditions
- All critical paths tested before deployment
- Comprehensive coverage helps prevent regressions

The position calculation tests were particularly important - found a few edge cases around floating point precision and LIFO behavior that could have caused issues in production.

