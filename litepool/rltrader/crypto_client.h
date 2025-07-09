#ifndef RLTRADER_CRYPTO_CLIENT_H
#define RLTRADER_CRYPTO_CLIENT_H

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/hmac.h>

namespace RLTrader
{
    namespace net   = boost::asio;
    namespace ssl   = boost::asio::ssl;
    namespace beast = boost::beast;
    namespace ws    = beast::websocket;

    using json = nlohmann::json;
    using tcp  = net::ip::tcp;

/* ---------------- simple threadsafe bounded queue ----------------- */
template<typename T>
class LockFreeQueue
{
public:
    bool push(const T& item)
    {
        std::lock_guard lk(m_);
        if (q_.size() >= kMax) return false;
        q_.push(item);
        return true;
    }
    bool push(T&& item)
    {
        std::lock_guard lk(m_);
        if (q_.size() >= kMax) return false;
        q_.push(std::move(item));
        return true;
    }
    bool pop(T& item)
    {
        std::lock_guard lk(m_);
        if (q_.empty()) return false;
        item = std::move(q_.front());
        q_.pop();
        return true;
    }
    bool empty() const
    {
        std::lock_guard lk(m_);
        return q_.empty();
    }
    void reset()
    {
        std::lock_guard lk(m_);
        std::queue<T>().swap(q_);
    }

private:
    static constexpr std::size_t kMax = 1024;
    mutable std::mutex           m_;
    std::queue<T>                q_;
};

/* ----------------------------- client ----------------------------- */
class CryptoClient
{
public:
    using websocket_stream = ws::stream<ssl::stream<beast::tcp_stream>>;

    CryptoClient(std::string api_key,
                 std::string api_secret,
                 std::string symbol,
                 std::string hedge_symbol = "");
    ~CryptoClient();

    void start();
    void stop();

    /* callbacks ----------------------------------------------------- */
    void set_orderbook_cb(std::function<void(const json&)> cb);
    void set_trade_cb    (std::function<void(const json&)> cb);
    void set_position_cb (std::function<void(const json&)> cb);
    void set_order_cb    (std::function<void(const json&)> cb);

    /* trading helpers ---------------------------------------------- */
    void place_order(const std::string& side,double price,double size,
                     const std::string& client_oid,bool hedge=false,
                     const std::string& type="LIMIT");
    void cancel_order(const std::string& order_id);
    void cancel_all_orders();
    void get_position();

private:
    /* connection helpers ------------------------------------------- */
    void setup_connections();
    void setup_public_ws();
    void setup_private_ws();
    void do_public_connect();
    void do_private_connect();

    void authenticate();
    void subscribe_public();
    void subscribe_private();

    /* messaging ----------------------------------------------------- */
    void send_public_msg (json&& j);
    void send_private_msg(json&& j);
    void write_next_public();
    void write_next_private();
    void do_public_read();
    void do_private_read();
    void handle_public_msg (const json& j);
    void handle_private_msg(const json& j);
    void handle_error(const std::string& where,
                      const beast::error_code& ec,
                      int http_status = 0);

    static std::string gen_id()
    {
        static std::atomic<uint64_t> id{0};
        return std::to_string(id++);
    }

    /* immutable ----------------------------------------------------- */
    const std::string api_key_;
    const std::string api_secret_;
    const std::string symbol_;
    const std::string hedge_symbol_;
    const std::string instance_id_;

    /* state --------------------------------------------------------- */
    std::atomic<bool> running_{false};
    bool public_connected_{false};
    bool private_connected_{false};
    bool public_writing_{false};
    bool private_writing_{false};
    int  retry_count_{0};

    std::unique_ptr<net::io_context> public_ioc_;
    std::unique_ptr<net::io_context> private_ioc_;
    std::unique_ptr<ssl::context>    ssl_ctx_;

    std::unique_ptr<tcp::resolver>   public_resolver_;
    std::unique_ptr<tcp::resolver>   private_resolver_;
    std::unique_ptr<websocket_stream> public_ws_;
    std::unique_ptr<websocket_stream> private_ws_;
    std::unique_ptr<net::steady_timer> public_timer_;
    std::unique_ptr<net::steady_timer> private_timer_;
    std::unique_ptr<std::thread>        public_thread_;
    std::unique_ptr<std::thread>        private_thread_;
    std::unique_ptr<net::executor_work_guard<net::io_context::executor_type>>
                                         public_work_;
    std::unique_ptr<net::executor_work_guard<net::io_context::executor_type>>
                                         private_work_;

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
