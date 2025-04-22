// deribit_client.h
#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/asio/executor_work_guard.hpp>
#include <atomic>
#include <thread>
#include <functional>
#include <memory>
#include <string>

namespace RLTrader {
    namespace beast = boost::beast;
    namespace websocket = beast::websocket;
    namespace net = boost::asio;
    namespace ssl = net::ssl;
    using tcp = net::ip::tcp;
    using json = nlohmann::json;
    using ssl_stream = beast::ssl_stream<tcp::socket>;
    using websocket_stream = websocket::stream<ssl_stream>;

    class DeribitClient {
    public:
        DeribitClient(std::string api_key,
                     std::string api_secret,
                     std::string symbol,
                     std::string hedge_symbol = "");
        ~DeribitClient();

        DeribitClient(const DeribitClient&) = delete;
        DeribitClient& operator=(const DeribitClient&) = delete;

        void start();
        void stop();

        // Trading operations
        void place_order(const std::string& side,
                        double price,
                        double size,
                        const std::string& label,
                        bool is_hedge = false,
                        const std::string& type = "limit");

        void cancel_order(const std::string& order_id);
        void cancel_all_by_label(const std::string& label);
        void cancel_all_orders();
        void get_position();

        // Callback setters
        void set_OrderBook_cb(std::function<void(const json&)> cb);
        void set_private_trade_cb(std::function<void(const json&)> cb);
        void set_position_cb(std::function<void(const json&)> cb);
        void set_order_cb(std::function<void(const json&)> cb);

    private:
        // Connection management
        void setup_connections();
        void setup_market_connection();
        void setup_trading_connection();
        void do_market_connect();
        void do_trading_connect();
        void authenticate();
        void subscribe_market_data();
        void subscribe_private_data();

        // Message handling
        void do_market_read();
        void do_trading_read();
        void handle_market_message(const json& msg);
        void handle_trading_message(const json& msg);
        void send_market_message(json&& msg);
        void send_trading_message(json&& msg);
        void handle_error(const std::string& context, const beast::error_code& ec);

        // Write operations
        void write_next_market_message();
        void write_next_trading_message();

        // Configuration
        const std::string api_key_;
        const std::string api_secret_;
        const std::string symbol_;
        const std::string hedge_symbol_;
        const size_t instance_id_;

        // IO and networking
        std::unique_ptr<net::io_context> market_ioc_;
        std::unique_ptr<net::io_context> trading_ioc_;
        std::unique_ptr<ssl::context> ctx_;
        std::unique_ptr<websocket_stream> market_ws_;
        std::unique_ptr<websocket_stream> trading_ws_;
        std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> market_work_;
        std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> trading_work_;
        std::unique_ptr<tcp::resolver> market_resolver_;
        std::unique_ptr<tcp::resolver> trading_resolver_;

        // State flags
        std::atomic<bool> market_connected_{false};
        std::atomic<bool> trading_connected_{false};
        std::atomic<bool> running_{false};
        std::atomic<bool> market_writing_{false};
        std::atomic<bool> trading_writing_{false};

        // Buffers
        beast::flat_buffer market_buffer_;
        beast::flat_buffer trading_buffer_;

        // Threads and timers
        std::unique_ptr<std::thread> market_thread_;
        std::unique_ptr<std::thread> trading_thread_;
        std::unique_ptr<net::steady_timer> market_timer_;
        std::unique_ptr<net::steady_timer> trading_timer_;

        // Callbacks
        std::function<void(const json&)> OrderBook_cb_;
        std::function<void(const json&)> private_trade_cb_;
        std::function<void(const json&)> position_cb_;
        std::function<void(const json&)> order_cb_;

        // Message queues
        static constexpr size_t QUEUE_SIZE = 1024;
        boost::lockfree::spsc_queue<json, boost::lockfree::capacity<QUEUE_SIZE>> market_out_queue_;
        boost::lockfree::spsc_queue<json, boost::lockfree::capacity<QUEUE_SIZE>> trading_out_queue_;

        static size_t generate_instance_id() {
            static std::atomic<size_t> counter{0};
            return counter++;
        }
    };
}
