#ifndef RLTRADER_CRYPTO_CLIENT_H
#define RLTRADER_CRYPTO_CLIENT_H

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <queue>
#include <condition_variable>

namespace RLTrader {

using net = boost::asio::ip;
using ssl = boost::asio::ssl;
using beast = boost::beast;
using websocket = beast::websocket;
using json = nlohmann::json;
using tcp = net::tcp;

// Lock-free queue for thread-safe message handling
template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue() = default;

    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) return false;
        queue_.push(std::move(item));
        lock.unlock();
        cond_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            cond_.wait(lock, [this] { return !queue_.empty(); });
        }
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

private:
    static constexpr size_t max_size_ = 1000;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

class CryptoClient {
public:
    using websocket_stream = websocket::stream<ssl::stream<beast::tcp_stream>>;

    CryptoClient(std::string api_key, std::string api_secret, std::string symbol, std::string hedge_symbol = "");
    ~CryptoClient();

    // Lifecycle
    void start();
    void stop();

    // Accessors / callbacks
    void set_orderbook_cb(std::function<void(const json&)> cb);
    void set_trade_cb(std::function<void(const json&)> cb);
    void set_position_cb(std::function<void(const json&)> cb);
    void set_order_cb(std::function<void(const json&)> cb);

    // Trading helpers
    void place_order(const std::string& side, double price, double size,
                     const std::string& client_oid, bool is_hedge = false,
                     const std::string& type = "LIMIT");
    void cancel_order(const std::string& order_id);
    void cancel_all_orders();
    void get_position();

private:
    // Lock hierarchy:
    // 1. public_mutex_ (for public_ws_, public_ioc_, public_resolver_, public_timer_, public_buffer_)
    // 2. private_mutex_ (for private_ws_, private_ioc_, private_resolver_, private_timer_, private_buffer_)
    // 3. queue mutexes (internal to LockFreeQueue, not shared)
    // Note: Never acquire multiple mutexes simultaneously to avoid deadlocks.
    //       ssl_ctx_ is immutable after construction and does not require locking.

    void setup_connections();
    void setup_public_ws();
    void setup_private_ws();
    void do_public_connect();
    void do_private_connect();
    void authenticate();
    void subscribe_public();
    void subscribe_private();
    void send_public_msg(json&& j);
    void send_private_msg(json&& j);
    void write_next_public();
    void write_next_private();
    void do_public_read();
    void do_private_read();
    void handle_public_msg(const json& j);
    void handle_private_msg(const json& j);
    void handle_error(const std::string& where, const beast::error_code& ec, int http_status = 0);

    static std::string gen_id() {
        static std::atomic<uint64_t> id{0};
        return std::to_string(id.fetch_add(1));
    }

    const std::string api_key_;
    const std::string api_secret_;
    const std::string symbol_;
    const std::string hedge_symbol_;
    const std::string instance_id_;

    std::atomic<bool> running_{false};
    bool public_connected_{false};
    bool private_connected_{false};
    bool public_writing_{false};
    bool private_writing_{false};
    int retry_count_{0};

    std::unique_ptr<net::io_context> public_ioc_;
    std::unique_ptr<net::io_context> private_ioc_;
    std::unique_ptr<ssl::context> ssl_ctx_;
    std::unique_ptr<tcp::resolver> public_resolver_;
    std::unique_ptr<tcp::resolver> private_resolver_;
    std::unique_ptr<websocket_stream> public_ws_;
    std::unique_ptr<websocket_stream> private_ws_;
    std::unique_ptr<net::steady_timer> public_timer_;
    std::unique_ptr<net::steady_timer> private_timer_;
    std::unique_ptr<std::thread> public_thread_;
    std::unique_ptr<std::thread> private_thread_;
    std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> public_work_;
    std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> private_work_;
    beast::flat_buffer public_buffer_;
    beast::flat_buffer private_buffer_;
    LockFreeQueue<json> public_q_;
    LockFreeQueue<json> private_q_;
    std::mutex public_mutex_;
    std::mutex private_mutex_;

    std::function<void(const json&)> orderbook_cb_;
    std::function<void(const json&)> trade_cb_;
    std::function<void(const json&)> position_cb_;
    std::function<void(const json&)> order_cb_;
};

} // namespace RLTrader

#endif // RLTRADER_CRYPTO_CLIENT_H
