// crypto_client.h
#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <functional>

namespace RLTrader {
    namespace beast     = boost::beast;
    namespace websocket = beast::websocket;
    namespace net       = boost::asio;
    namespace ssl       = net::ssl;
    using     tcp       = net::ip::tcp;
    using     json      = nlohmann::json;
    using     ssl_stream      = beast::ssl_stream<tcp::socket>;
    using     websocket_stream = websocket::stream<ssl_stream>;

    class CryptoClient {
    public:
        CryptoClient(std::string api_key,
                     std::string api_secret,
                     std::string symbol,
                     std::string hedge_symbol = "");
        ~CryptoClient();

        CryptoClient(const CryptoClient&)            = delete;
        CryptoClient& operator=(const CryptoClient&) = delete;

        void start();
        void stop();

        /* Trading helpers -------------------------------------------------- */
        void place_order(const std::string& side,
                         double price,
                         double size,
                         const std::string& client_oid,
                         bool is_hedge = false,
                         const std::string& type = "LIMIT");

        void cancel_order(const std::string& order_id);
        void cancel_all_orders();
        void get_position();

        /* User-supplied callbacks ------------------------------------------ */
        void set_orderbook_cb (std::function<void(const json&)> cb);
        void set_trade_cb     (std::function<void(const json&)> cb);
        void set_position_cb  (std::function<void(const json&)> cb);
        void set_order_cb     (std::function<void(const json&)> cb);

    private:
        /* Internal helpers ------------------------------------------------- */
        void setup_connections();
        void setup_public_ws();
        void setup_private_ws();

        void do_public_connect();
        void do_private_connect();

        void authenticate();
        void subscribe_public();
        void subscribe_private();

        void do_public_read();
        void do_private_read();

        void handle_public_msg (const json& j);
        void handle_private_msg(const json& j);

        void send_public_msg (json&& j);
        void send_private_msg(json&& j);

        void write_next_public ();
        void write_next_private();

        void handle_error(const std::string& where, const beast::error_code& ec);

        /* Configuration ---------------------------------------------------- */
        const std::string api_key_;
        const std::string api_secret_;
        const std::string symbol_;
        const std::string hedge_symbol_;
        const size_t      instance_id_;

        /* Networking ------------------------------------------------------- */
        std::unique_ptr<net::io_context> public_ioc_;
        std::unique_ptr<net::io_context> private_ioc_;
        std::unique_ptr<ssl::context>    ssl_ctx_;

        std::unique_ptr<websocket_stream> public_ws_;
        std::unique_ptr<websocket_stream> private_ws_;

        std::unique_ptr<tcp::resolver> public_resolver_;
        std::unique_ptr<tcp::resolver> private_resolver_;

        beast::flat_buffer public_buffer_;
        beast::flat_buffer private_buffer_;

        std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> public_work_;
        std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> private_work_;

        std::unique_ptr<std::thread> public_thread_;
        std::unique_ptr<std::thread> private_thread_;

        std::unique_ptr<net::steady_timer> public_timer_;
        std::unique_ptr<net::steady_timer> private_timer_;

        std::atomic<bool> running_{false};
        std::atomic<bool> public_connected_{false};
        std::atomic<bool> private_connected_{false};
        std::atomic<bool> public_writing_{false};
        std::atomic<bool> private_writing_{false};

        /* User callbacks --------------------------------------------------- */
        std::function<void(const json&)> orderbook_cb_;
        std::function<void(const json&)> trade_cb_;
        std::function<void(const json&)> position_cb_;
        std::function<void(const json&)> order_cb_;

        /* Outgoing queues -------------------------------------------------- */
        static constexpr size_t QUEUE_SIZE = 1024;
        boost::lockfree::spsc_queue<json, boost::lockfree::capacity<QUEUE_SIZE>> public_q_;
        boost::lockfree::spsc_queue<json, boost::lockfree::capacity<QUEUE_SIZE>> private_q_;

        static size_t gen_id() { static std::atomic<size_t> c{0}; return c++; }
    };
}
