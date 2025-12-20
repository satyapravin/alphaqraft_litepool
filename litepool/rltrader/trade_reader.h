// trade_reader.h
#pragma once
#include "order.h"
#include <vector>
#include <optional>
#include <fstream>
#include <string>

namespace RLTrader {
    struct Trade {
        long long timestamp;
        double price;
        double size;
        OrderSide side;  // BUY or SELL (aggressor side)
    };

    class TradeReader {
    private:
        std::string filename;  // Store filename for manual seeking
        std::ifstream file_stream;  // Direct file access for parsing string columns
        int start_read;  // Store start_read for creating new readers
        long long current_book_timestamp;  // Last book timestamp we processed
        long long last_processed_trade_ts;  // Last trade timestamp we processed
        long long target_start_timestamp;  // Timestamp to sync to on reset
        
        // Buffer for peeking at next trade without consuming
        std::optional<Trade> next_trade_buffer;
        
        // Parse trade CSV row from line string
        // Format: exchange,symbol,timestamp,local_timestamp,id,side,price,amount
        Trade parseTradeLine(const std::string& line);
        
        // Load next trade into buffer (peek ahead)
        void loadNextTrade();
        
        // Seek to specific timestamp in trade CSV (for synchronization)
        void seekToTimestamp(long long target_ts);
        
        // Reset file stream to beginning
        void resetFileStream();
        
    public:
        TradeReader(const std::string& filename, int start_read);
        ~TradeReader();
        
        // Get trades up to book_timestamp (synchronized with book reader)
        // Only advances trade reader when book_timestamp > current_book_timestamp
        std::vector<Trade> getRecentTrades(long long book_timestamp);
        
        // Reset and sync to book reader's starting timestamp
        void reset(long long book_start_timestamp);
        
        bool hasNext();
    };
}

