#ifndef CRYPTO_CLIENT_H
#define CRYPTO_CLIENT_H

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace RLTrader {

class CryptoClient {
public:
    using tcp = boost::asio::ip::tcp;
    using ssl_stream = boost::asio::ssl::stream<tcp::socket>;
    using websocket_stream = boost::beast::websocket::stream<ssl_stream>;
    using json = nlohmann::json;

    CryptoClient(std::string api_key,
                std::string api_secret,
                std::string symbol,
                std::string hedge_symbol = "");
    ~CryptoClient();

    // Lifecycle management
    void start();
    void stop();

    // Callback setters
    void set_orderbook_cb(std::function<void(const json&)> cb);
    void set_trade_cb(std::function<void(const json&)> cb);
    void set_position_cb(std::function<void(const json&)> cb);
    void set_order_cb(std::function<void(const json&)> cb);

    // Trading operations
    void place_order(const std::string& side, double price, double size,
                    const std::string& client_oid = "", bool hedge = false,
                    const std::string& type = "LIMIT");
    void cancel_order(const std::string& order_id);
    void cancel_all_orders();
    void get_position();

private:
    // Connection management
    void setup_public_ws();
    void setup_private_ws();
    void do_public_connect();
    void do_private_connect();

    // Ping management
    void start_public_ping();
    void start_private_ping();

    // Authentication & subscriptions
    void authenticate();
    void subscribe_public();
    void subscribe_private();

    // Messaging
    void send_public_msg(json&& j);
    void send_private_msg(json&& j);

    // Read loops
    void do_public_read();
    void do_private_read();

    // Message handling
    void handle_public_msg(const json& j);
    void handle_private_msg(const json& j);
    void handle_public_error(const std::string& where, boost::beast::error_code ec);
    void handle_private_error(const std::string& where, boost::beast::error_code ec);

    // Utility
    std::string gen_id() const;
    std::string build_payload(const std::string& method, const std::string& id,
                             const std::string& api_key, const json& params, long nonce) const;

    // API credentials
    const std::string api_key_;
    const std::string api_secret_;
    const std::string symbol_;
    const std::string hedge_symbol_;
    const std::string instance_id_;

    // Network components
    std::unique_ptr<boost::asio::io_context> public_ioc_;
    std::unique_ptr<boost::asio::io_context> private_ioc_;
    std::unique_ptr<boost::asio::ssl::context> ssl_ctx_;

    // Public stream components
    std::mutex public_mutex_;
    std::unique_ptr<tcp::resolver> public_resolver_;
    std::unique_ptr<websocket_stream> public_ws_;
    boost::beast::flat_buffer public_buffer_;
    std::unique_ptr<boost::asio::steady_timer> public_ping_timer_;
    std::unique_ptr<boost::asio::steady_timer> public_heartbeat_timer_;
    std::unique_ptr<boost::asio::executor_work_guard<
        boost::asio::io_context::executor_type>> public_work_;

    // Private stream components
    std::mutex private_mutex_;
    std::unique_ptr<tcp::resolver> private_resolver_;
    std::unique_ptr<websocket_stream> private_ws_;
    boost::beast::flat_buffer private_buffer_;
    std::unique_ptr<boost::asio::steady_timer> private_ping_timer_;
    std::unique_ptr<boost::asio::steady_timer> private_heartbeat_timer_;
    std::unique_ptr<boost::asio::executor_work_guard<
        boost::asio::io_context::executor_type>> private_work_;

    // Threads
    std::thread public_thread_;
    std::thread private_thread_;

    // State flags
    std::atomic<bool> running_{false};
    std::atomic<bool> public_connected_{false};
    std::atomic<bool> private_connected_{false};

    // Callbacks
    std::function<void(const json&)> orderbook_cb_;
    std::function<void(const json&)> trade_cb_;
    std::function<void(const json&)> position_cb_;
    std::function<void(const json&)> order_cb_;

    // Constants
    static constexpr const char* CR_PUBLIC_HOST = "stream.crypto.com";
    static constexpr const char* CR_PRIVATE_HOST = "stream.crypto.com";
    static constexpr const char* CR_PUBLIC_PATH = "/exchange/v1/market";
    static constexpr const char* CR_PRIVATE_PATH = "/exchange/v1/user";
    static constexpr const char* CR_SSL_PORT = "443";
    static constexpr const char* USER_AGENT = "AlphaQraft_Trading";
};

} // namespace RLTrader

#endif // CRYPTO_CLIENT_H
