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

#include "trade_reader.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cctype>
#include <glog/logging.h>

using namespace RLTrader;

TradeReader::TradeReader(
    const std::string& filename, 
    int start_read
) : filename(filename),
    start_read(start_read),
    current_book_timestamp(0),
    last_processed_trade_ts(0),
    target_start_timestamp(0) {
    resetFileStream();
}

TradeReader::~TradeReader() {
    if (file_stream.is_open()) {
        file_stream.close();
    }
}

void TradeReader::resetFileStream() {
    if (file_stream.is_open()) {
        file_stream.close();
    }
    file_stream.open(filename);
    if (!file_stream.is_open()) {
        LOG(WARNING) << "Could not open trade file: " << filename;
        return;
    }
    
    // Skip header line
    std::string header;
    if (!std::getline(file_stream, header)) {
        LOG(WARNING) << "Could not read header from trade file: " << filename;
    }
}

void TradeReader::reset(long long book_start_timestamp) {
    // CRITICAL: Sync trade reader to book reader's starting timestamp
    target_start_timestamp = book_start_timestamp;
    
    // Reset file stream
    resetFileStream();
    
    // Find the row in trade CSV with timestamp >= book_start_timestamp
    seekToTimestamp(book_start_timestamp);
    
    current_book_timestamp = 0;  // Will be set on first getRecentTrades() call
    last_processed_trade_ts = 0;
    next_trade_buffer.reset();
}

void TradeReader::seekToTimestamp(long long target_ts) {
    // Strategy: Manually scan the CSV file from beginning until we find first row with timestamp >= target_ts
    // Format: exchange,symbol,timestamp,local_timestamp,id,side,price,amount
    
    next_trade_buffer.reset();
    
    // File stream should already be open and at header (after resetFileStream)
    if (!file_stream.is_open()) {
        resetFileStream();
    }
    
    // Reset to beginning (after header)
    file_stream.clear();
    file_stream.seekg(0, std::ios::beg);
    std::string header;
    std::getline(file_stream, header);  // Skip header
    
    // Scan through file to find target timestamp
    std::string line;
    while (std::getline(file_stream, line)) {
        Trade trade = parseTradeLine(line);
        if (trade.timestamp == 0) continue;  // Skip malformed lines
        
        if (trade.timestamp >= target_ts) {
            // Found it - buffer this trade
            next_trade_buffer = trade;
            return;
        }
    }
    
    // If we reach here, all trades are before target_ts
    // next_trade_buffer remains empty
}

void TradeReader::loadNextTrade() {
    if (file_stream.is_open() && !next_trade_buffer.has_value()) {
        std::string line;
        if (std::getline(file_stream, line)) {
            Trade trade = parseTradeLine(line);
            if (trade.timestamp > 0) {  // Valid trade
                next_trade_buffer = trade;
            }
        }
    }
}

std::vector<Trade> TradeReader::getRecentTrades(long long book_timestamp) {
    std::vector<Trade> recent;
    
    // First call after reset: ensure we're synced to starting timestamp
    if (current_book_timestamp == 0 && target_start_timestamp > 0) {
        // If we don't have a buffered trade yet, we need to scan to target_start_timestamp
        if (!next_trade_buffer.has_value()) {
            // Continue scanning from current position until we find target_start_timestamp
            std::string line;
            while (std::getline(file_stream, line)) {
                Trade trade = parseTradeLine(line);
                if (trade.timestamp == 0) continue;  // Skip malformed
                
                if (trade.timestamp >= target_start_timestamp) {
                    // Found starting point - buffer this trade
                    next_trade_buffer = trade;
                    break;
                }
            }
        }
    }
    
    // CRITICAL: Only advance trade reader if book timestamp has advanced
    if (book_timestamp <= current_book_timestamp) {
        // Book hasn't advanced, return empty
        return recent;
    }
    
    // Ensure we have next trade buffered
    loadNextTrade();
    
    // Collect all trades up to book_timestamp
    while (next_trade_buffer.has_value()) {
        long long trade_ts = next_trade_buffer->timestamp;
        
        if (trade_ts > book_timestamp) {
            // Trade is in the future - don't consume it yet
            break;
        }
        
        // Trade is at or before book timestamp - process it
        if (trade_ts > last_processed_trade_ts) {
            recent.push_back(*next_trade_buffer);
            last_processed_trade_ts = trade_ts;
        }
        
        // Clear buffer and load next trade
        next_trade_buffer.reset();
        loadNextTrade();
    }
    
    // Update current book timestamp
    current_book_timestamp = book_timestamp;
    
    return recent;
}

Trade TradeReader::parseTradeLine(const std::string& line) {
    // Format: exchange,symbol,timestamp,local_timestamp,id,side,price,amount
    Trade trade;
    trade.timestamp = 0;  // Default to invalid
    
    std::istringstream lineStream(line);
    std::string cell;
    
    // Skip exchange (1st column)
    if (!std::getline(lineStream, cell, ',')) return trade;
    
    // Skip symbol (2nd column)
    if (!std::getline(lineStream, cell, ',')) return trade;
    
    // Get timestamp (3rd column)
    if (!std::getline(lineStream, cell, ',')) return trade;
        trade.timestamp = std::stoll(cell);
    
    // Skip local_timestamp (4th column)
    if (!std::getline(lineStream, cell, ',')) return trade;
    
    // Skip id (5th column)
    if (!std::getline(lineStream, cell, ',')) return trade;
    
    // Get side (6th column) - string: "buy" or "sell"
    if (!std::getline(lineStream, cell, ',')) return trade;
    // Convert to lowercase and compare
    std::string side_str = cell;
    std::transform(side_str.begin(), side_str.end(), side_str.begin(), ::tolower);
    if (side_str == "buy") {
        trade.side = OrderSide::BUY;
    } else if (side_str == "sell") {
        trade.side = OrderSide::SELL;
    } else {
        return trade;  // Invalid side
    }
    
    // Get price (7th column)
    if (!std::getline(lineStream, cell, ',')) return trade;
        trade.price = std::stod(cell);
    
    // Get amount (8th column) - this is the trade size
    if (!std::getline(lineStream, cell, ',')) return trade;
        trade.size = std::stod(cell);
    
    return trade;
}

bool TradeReader::hasNext() {
    return next_trade_buffer.has_value() || (file_stream.is_open() && file_stream.good());
}

