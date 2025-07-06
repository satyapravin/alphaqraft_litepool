#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <nlohmann/json.hpp>

namespace RLTrader {
    namespace beast = boost::beast;
    namespace net = boost::asio;
    namespace ssl = net::ssl;
    namespace websocket = beast::websocket;
    using json = nlohmann::json;
    using websocket_stream = websocket::stream<ssl::stream<beast::tcp_stream>>;

    class CryptoClient {
    public:
        CryptoClient(std::string api_key, std::string api_secret, std::string symbol, std::string hedge_symbol = {});
        ~CryptoClient();

        void start();
        void stop();

        void set_orderbook_cb(std::function<void(const json&)> cb);
        void set_trade_cb(std::function<void(const json&)> cb);
        void set_position_cb(std::function<void(const json&)> cb);
        void set_order_cb(std::function<void(const json&)> cb);

        void place_order(const std::string& side, double price, double size,
                         const std::string& client_oid, bool is_hedge, const std::string& type);
        void cancel_order(const std::string& order_id);
        void cancel_all_orders();
        void get_position();

    private:
        using tcp = boost::asio::ip::tcp;

        template<typename T, size_t N = 16> class Queue {
            std::array<T, N> items;
            size_t head = 0, tail = 0, sz = 0;

        public:
            bool push(T&& t) {
                if (sz >= N) return false;
                items[tail] = std::move(t);
                tail = (tail + 1) % N;
                ++sz;
                return true;
            }
            bool pop(T& t) {
                if (sz == 0) return false;
                t = std::move(items[head]);
                head = (head + 1) % N;
                --sz;
                return true;
            }
            void reset() { head = tail = sz = 0; }
            bool empty() const { return sz == 0; }
        };

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
        void handle_error(const std::string& where, const beast::error_code& ec, int http_status = -1);

        static size_t gen_id() {
            static std::atomic<size_t> id{0};
            return id++;
        }

        std::string api_key_;
        std::string api_secret_;
        std::string symbol_;
        std::string hedge_symbol_;
        size_t instance_id_;
        std::atomic<bool> running_{false};
        std::atomic<bool> public_connected_{false};
        std::atomic<bool> private_connected_{false};
        std::atomic<bool> public_writing_{false};
        std::atomic<bool> private_writing_{false};
        std::atomic<size_t> retry_count_{0};
        std::unique_ptr<net::io_context> public_ioc_;
        std::unique_ptr<net::io_context> private_ioc_;
        std::unique_ptr<ssl::context> ssl_ctx_;
        std::unique_ptr<tcp::resolver> public_resolver_;
        std::unique_ptr<tcp::resolver> private_resolver_;
        std::unique_ptr<websocket_stream> public_ws_;
        std::unique_ptr<websocket_stream> private_ws_;
        std::unique_ptr<std::thread> public_thread_;
        std::unique_ptr<std::thread> private_thread_;
        std::unique_ptr<net::steady_timer> public_timer_;
        std::unique_ptr<net::steady_timer> private_timer_;
        std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> public_work_;
        std::unique_ptr<boost::asio::executor_work_guard<net::io_context::executor_type>> private_work_;
        beast::flat_buffer public_buffer_;
        beast::flat_buffer private_buffer_;
        Queue<json> public_q_;
        Queue<json> private_q_;
        std::function<void(const json&)> orderbook_cb_;
        std::function<void(const json&)> trade_cb_;
        std::function<void(const json&)> position_cb_;
        std::function<void(const json&)> order_cb_;
        std::mutex ws_mutex_;
    };
}
