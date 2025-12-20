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

#ifndef CRYPTO_CLIENT_H
#define CRYPTO_CLIENT_H

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <queue>
#include <condition_variable>

namespace RLTrader {

using json = nlohmann::json;
using tcp = boost::asio::ip::tcp;
using websocket_stream = boost::beast::websocket::stream<
    boost::asio::ssl::stream<boost::beast::tcp_stream>>;

class CryptoClient {
public:
    CryptoClient(std::string api_key, std::string api_secret,
                 std::string symbol, std::string hedge_symbol = "");
    ~CryptoClient();

    // Lifecycle
    void start();
    void stop();

    // Callbacks
    void set_orderbook_cb(std::function<void(const json&)> cb);
    void set_trade_cb(std::function<void(const json&)> cb);
    void set_position_cb(std::function<void(const json&)> cb);
    void set_order_cb(std::function<void(const json&)> cb);

    // Trading operations
    void place_order(const std::string& side, double price, double size,
                     const std::string& client_oid, bool hedge = false,
                     const std::string& type = "LIMIT");
    void cancel_order(const std::string& order_id);
    void cancel_all_orders();
    void get_position();

private:
    // Setup
    void setup_public_ws();
    void setup_private_ws();

    // Connection
    void do_public_connect();
    void do_private_connect();

    // Ping
    void start_public_ping();
    void start_private_ping();

    // Read
    void do_public_read();
    void do_private_read();

    // Message handling
    void send_public_msg(json&& j);
    void send_private_msg(json&& j);
    void run_private_write_thread();
    void handle_public_msg(const json& j);
    void handle_private_msg(const json& j);
    void handle_public_error(const std::string& where, boost::beast::error_code ec);
    void handle_private_error(const std::string& where, boost::beast::error_code ec);

    // Subscriptions
    void subscribe_public();
    void subscribe_private();
    void authenticate();

    // Utilities
    std::string gen_id() const;
    std::string build_payload(const std::string& method, const int id,
                              const std::string& api_key, const json& params, long nonce) const;

    // Members
    std::string api_key_;
    std::string api_secret_;
    std::string symbol_;
    std::string hedge_symbol_;
    std::string instance_id_;

    std::atomic<bool> running_{false};
    bool public_connected_{false};
    bool private_connected_{false};
    bool private_authenticated_{false};

    // Public stream resources
    std::unique_ptr<boost::asio::io_context> public_ioc_;
    std::unique_ptr<websocket_stream> public_ws_;
    std::unique_ptr<tcp::resolver> public_resolver_;
    std::unique_ptr<boost::asio::steady_timer> public_ping_timer_;
    std::unique_ptr<boost::asio::steady_timer> public_heartbeat_timer_;
    std::unique_ptr<boost::asio::executor_work_guard<
        boost::asio::io_context::executor_type>> public_work_;
    boost::beast::multi_buffer public_buffer_;
    std::mutex public_mutex_;
    std::thread public_thread_;
    std::unique_ptr<boost::asio::strand<boost::asio::io_context::executor_type>> public_write_strand_;

    // Private stream resources
    std::unique_ptr<boost::asio::io_context> private_ioc_;
    std::unique_ptr<websocket_stream> private_ws_;
    std::unique_ptr<tcp::resolver> private_resolver_;
    std::unique_ptr<boost::asio::steady_timer> private_ping_timer_;
    std::unique_ptr<boost::asio::steady_timer> private_heartbeat_timer_;
    std::unique_ptr<boost::asio::executor_work_guard<
        boost::asio::io_context::executor_type>> private_work_;
    boost::beast::multi_buffer private_buffer_;
    std::mutex private_mutex_;
    std::thread private_thread_;
    std::unique_ptr<boost::asio::strand<boost::asio::io_context::executor_type>> private_write_strand_;

    // Thread-safe queue for private messages
    std::queue<json> private_msg_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread private_write_thread_;
    std::atomic<bool> write_thread_running_{false};

    // SSL context
    std::unique_ptr<boost::asio::ssl::context> ssl_ctx_;

    // Callbacks
    std::function<void(const json&)> orderbook_cb_;
    std::function<void(const json&)> trade_cb_;
    std::function<void(const json&)> position_cb_;
    std::function<void(const json&)> order_cb_;
};

} // namespace RLTrader

#endif // CRYPTO_CLIENT_H
