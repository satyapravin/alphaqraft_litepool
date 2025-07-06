
#include "crypto_client.h"
#include <boost/asio/connect.hpp>
#include <boost/asio/post.hpp>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace RLTrader {

constexpr const char *CR_PUBLIC_HOST  = "uat-stream.3ona.co";
constexpr const char *CR_PRIVATE_HOST = "uat-stream.3ona.co";
//constexpr const char *CR_PUBLIC_HOST  = "stream.crypto.com";
//constexpr const char *CR_PRIVATE_HOST = "stream.crypto.com";
constexpr const char *CR_PUBLIC_PATH  = "/exchange/v1/market";
constexpr const char *CR_PRIVATE_PATH = "/exchange/v1/user";
constexpr const char *CR_SSL_PORT     = "443";
constexpr const char *USER_AGENT     = "RLTrader/1.0 (Crypto.com V1 API Client)";

static constexpr std::chrono::milliseconds CONNECT_DELAY{30000}; // 30s to avoid rate limits

/* --------------------------------------------------------------------- */
CryptoClient::CryptoClient(std::string api_key,
                           std::string api_secret,
                           std::string symbol,
                           std::string hedge_symbol)
    : api_key_(std::move(api_key)),
      api_secret_(std::move(api_secret)),
      symbol_(std::move(symbol)),
      hedge_symbol_(hedge_symbol.empty() ? symbol_ : std::move(hedge_symbol)),
      instance_id_(gen_id()) {
    std::cout << "[CryptoClient#" << instance_id_ << "] constructed\n";
}

CryptoClient::~CryptoClient() { stop(); }

/* ------------------------- life-cycle -------------------------------- */
void CryptoClient::start() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering start\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (running_.exchange(true, std::memory_order_acquire)) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Already running, skipping start\n";
        return;
    }

    // Initialize fresh resources
    public_ioc_  = std::make_unique<net::io_context>();
    private_ioc_ = std::make_unique<net::io_context>();
    ssl_ctx_     = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
    ssl_ctx_->set_default_verify_paths();
    ssl_ctx_->set_verify_mode(ssl::verify_peer);

    public_resolver_  = std::make_unique<tcp::resolver>(*public_ioc_);
    private_resolver_ = std::make_unique<tcp::resolver>(*private_ioc_);

    public_work_  = std::make_unique<boost::asio::executor_work_guard<net::io_context::executor_type>>(public_ioc_->get_executor());
    private_work_ = std::make_unique<boost::asio::executor_work_guard<net::io_context::executor_type>>(private_ioc_->get_executor());

    // Clear buffers and queues
    public_buffer_.clear();
    private_buffer_.clear();
    public_q_.reset();
    private_q_.reset();

    std::cout << "[CryptoClient#" << instance_id_ << "] Setting up connections\n";
    setup_connections();

    std::cout << "[CryptoClient#" << instance_id_ << "] Starting threads\n";
    public_thread_  = std::make_unique<std::thread>([this] {
        std::cout << "[CryptoClient#" << instance_id_ << "] Public thread started\n";
        public_ioc_->run();
        std::cout << "[CryptoClient#" << instance_id_ << "] Public thread exited\n";
    });
    private_thread_ = std::make_unique<std::thread>([this] {
        std::cout << "[CryptoClient#" << instance_id_ << "] Private thread started\n";
        private_ioc_->run();
        std::cout << "[CryptoClient#" << instance_id_ << "] Private thread exited\n";
    });

    std::cout << "[CryptoClient#" << instance_id_ << "] Started\n";
}

void CryptoClient::stop() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering stop\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!running_.exchange(false, std::memory_order_release)) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Already stopped, skipping stop\n";
        return;
    }

    boost::system::error_code ec;
    // Cancel pending timers
    if (public_timer_) {
        public_timer_->cancel(ec);
        if (ec && ec != boost::asio::error::operation_aborted) {
            std::cerr << "[CryptoClient#" << instance_id_ << "] Public timer cancel: " << ec.message() << '\n';
        }
    }
    if (private_timer_) {
        private_timer_->cancel(ec);
        if (ec && ec != boost::asio::error::operation_aborted) {
            std::cerr << "[CryptoClient#" << instance_id_ << "] Private timer cancel: " << ec.message() << '\n';
        }
    }

    // Close WebSocket connections
    if (public_ws_ && public_connected_) {
        public_ws_->close(websocket::close_code::normal, ec);
        if (ec) std::cerr << "[CryptoClient#" << instance_id_ << "] Public WS close: " << ec.message() << '\n';
        public_connected_ = false;
    }
    if (private_ws_ && private_connected_) {
        private_ws_->close(websocket::close_code::normal, ec);
        if (ec) std::cerr << "[CryptoClient#" << instance_id_ << "] Private WS close: " << ec.message() << '\n';
        private_connected_ = false;
    }

    // Reset work guards and stop io_contexts
    if (public_work_) {
        public_work_.reset();
        public_ioc_->stop();
    }
    if (private_work_) {
        private_work_.reset();
        private_ioc_->stop();
    }

    // Join threads
    if (public_thread_ && public_thread_->joinable()) {
        std::cout << "[CryptoClient#" << instance_id_ << "] Joining public thread\n";
        public_thread_->join();
    }
    if (private_thread_ && private_thread_->joinable()) {
        std::cout << "[CryptoClient#" << instance_id_ << "] Joining private thread\n";
        private_thread_->join();
    }

    // Reset all resources to ensure no dangling pointers
    public_ws_.reset();
    private_ws_.reset();
    public_resolver_.reset();
    private_resolver_.reset();
    public_timer_.reset();
    private_timer_.reset();
    public_ioc_.reset();
    private_ioc_.reset();
    ssl_ctx_.reset();
    public_connected_ = false;
    private_connected_ = false;
    public_writing_ = false;
    private_writing_ = false;
    retry_count_ = 0;

    std::cout << "[CryptoClient#" << instance_id_ << "] Stopped\n";
}

/* ---------------------- connection helpers --------------------------- */
void CryptoClient::setup_connections() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering setup_connections\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    setup_public_ws();
    setup_private_ws();
}

void CryptoClient::setup_public_ws() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering setup_public_ws\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!public_ioc_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Public io_context is null\n";
        return;
    }
    public_ws_ = std::make_unique<websocket_stream>(*public_ioc_, *ssl_ctx_);
    public_ws_->set_option(websocket::stream_base::decorator(
        [](websocket::request_type& req) {
            req.set(beast::http::field::user_agent, USER_AGENT);
            req.set(beast::http::field::host, CR_PUBLIC_HOST);
        }));
    public_timer_ = std::make_unique<net::steady_timer>(*public_ioc_);
    public_timer_->expires_after(CONNECT_DELAY);
    std::cout << "[CryptoClient#" << instance_id_ << "] Scheduling public timer\n";
    public_timer_->async_wait([this](auto ec) {
        std::cout << "[CryptoClient#" << instance_id_ << "] Public timer fired\n";
        if (!ec && running_) do_public_connect();
        else if (ec && ec != boost::asio::error::operation_aborted) {
            std::cerr << "[CryptoClient#" << instance_id_ << "] Public timer error: " << ec.message() << '\n';
        }
    });
}

void CryptoClient::setup_private_ws() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering setup_private_ws\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!private_ioc_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Private io_context is null\n";
        return;
    }
    private_ws_ = std::make_unique<websocket_stream>(*private_ioc_, *ssl_ctx_);
    private_ws_->set_option(websocket::stream_base::decorator(
        [](websocket::request_type& req) {
            req.set(beast::http::field::user_agent, USER_AGENT);
            req.set(beast::http::field::host, CR_PRIVATE_HOST);
        }));
    private_timer_ = std::make_unique<net::steady_timer>(*private_ioc_);
    private_timer_->expires_after(CONNECT_DELAY);
    std::cout << "[CryptoClient#" << instance_id_ << "] Scheduling private timer\n";
    private_timer_->async_wait([this](auto ec) {
        std::cout << "[CryptoClient#" << instance_id_ << "] Private timer fired\n";
        if (!ec && running_) do_private_connect();
        else if (ec && ec != boost::asio::error::operation_aborted) {
            std::cerr << "[CryptoClient#" << instance_id_ << "] Private timer error: " << ec.message() << '\n';
        }
    });
}

void CryptoClient::do_public_connect() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering do_public_connect\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!public_resolver_ || !public_ws_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Public resolver or WebSocket is null\n";
        return;
    }
    std::cout << "[CryptoClient#" << instance_id_ << "] Initiating public WS connection\n";
    public_resolver_->async_resolve(CR_PUBLIC_HOST, CR_SSL_PORT,
        [this](auto ec, tcp::resolver::results_type res) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Public resolve completed\n";
            if (ec) return handle_error("Public resolve", ec);

            // Create a Beast TCP stream for connection
            auto stream = std::make_shared<beast::tcp_stream>(*public_ioc_);
            beast::get_lowest_layer(*stream).async_connect(res,
                [this, stream](auto ec, tcp::endpoint) {
                    std::cout << "[CryptoClient#" << instance_id_ << "] Public TCP connect completed\n";
                    if (ec) return handle_error("Public connect", ec);

                    // Wrap stream with SSL stream
                    auto ssl_stream = std::make_shared<ssl::stream<beast::tcp_stream>>(std::move(*stream), *ssl_ctx_);
                    ssl_stream->async_handshake(ssl::stream_base::client,
                        [this, ssl_stream](auto ec) {
                            std::cout << "[CryptoClient#" << instance_id_ << "] Public SSL handshake completed\n";
                            if (ec) return handle_error("Public SSL", ec);

                            // Assign SSL stream to WebSocket stream
                            public_ws_ = std::make_unique<websocket_stream>(std::move(*ssl_stream));
                            public_ws_->set_option(websocket::stream_base::decorator(
                                [](websocket::request_type& req) {
                                    req.set(beast::http::field::user_agent, USER_AGENT);
                                    req.set(beast::http::field::host, CR_PUBLIC_HOST);
                                }));

                            auto response = std::make_shared<websocket::response_type>();
                            std::cout << "[CryptoClient#" << instance_id_ << "] Starting public WS handshake\n";
                            public_ws_->async_handshake(*response, CR_PUBLIC_HOST, CR_PUBLIC_PATH,
                                [this, response](auto ec) {
                                    std::cout << "[CryptoClient#" << instance_id_ << "] Public WS handshake completed\n";
                                    if (ec) {
                                        std::cerr << "[CryptoClient#" << instance_id_ << "] Public WS handshake failed: "
                                                  << ec.message() << ", HTTP Status: " << response->result_int()
                                                  << ", Reason: " << response->reason() << '\n';
                                        handle_error("Public WS", ec, response->result_int());
                                        return;
                                    }
                                    public_connected_ = true;
                                    std::cout << "[CryptoClient#" << instance_id_ << "] Public WS connected\n";
                                    public_timer_->expires_after(CONNECT_DELAY);
                                    public_timer_->async_wait([this](auto ec) {
                                        std::cout << "[CryptoClient#" << instance_id_ << "] Public subscription timer fired\n";
                                        if (!ec && running_) {
                                            subscribe_public();
                                            do_public_read();
                                        }
                                    });
                                });
                        });
                });
        });
}

void CryptoClient::do_private_connect() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering do_private_connect\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!private_resolver_ || !private_ws_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Private resolver or WebSocket is null\n";
        return;
    }
    std::cout << "[CryptoClient#" << instance_id_ << "] Initiating private WS connection\n";
    private_resolver_->async_resolve(CR_PRIVATE_HOST, CR_SSL_PORT,
        [this](auto ec, tcp::resolver::results_type res) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Private resolve completed\n";
            if (ec) return handle_error("Private resolve", ec);

            // Create a Beast TCP stream for connection
            auto stream = std::make_shared<beast::tcp_stream>(*private_ioc_);
            beast::get_lowest_layer(*stream).async_connect(res,
                [this, stream](auto ec, tcp::endpoint) {
                    std::cout << "[CryptoClient#" << instance_id_ << "] Private TCP connect completed\n";
                    if (ec) return handle_error("Private connect", ec);

                    // Wrap stream with SSL stream
                    auto ssl_stream = std::make_shared<ssl::stream<beast::tcp_stream>>(std::move(*stream), *ssl_ctx_);
                    ssl_stream->async_handshake(ssl::stream_base::client,
                        [this, ssl_stream](auto ec) {
                            std::cout << "[CryptoClient#" << instance_id_ << "] Private SSL handshake completed\n";
                            if (ec) return handle_error("Private SSL", ec);

                            // Assign SSL stream to WebSocket stream
                            private_ws_ = std::make_unique<websocket_stream>(std::move(*ssl_stream));
                            private_ws_->set_option(websocket::stream_base::decorator(
                                [](websocket::request_type& req) {
                                    req.set(beast::http::field::user_agent, USER_AGENT);
                                    req.set(beast::http::field::host, CR_PRIVATE_HOST);
                                }));

                            auto response = std::make_shared<websocket::response_type>();
                            std::cout << "[CryptoClient#" << instance_id_ << "] Starting private WS handshake\n";
                            private_ws_->async_handshake(*response, CR_PRIVATE_HOST, CR_PRIVATE_PATH,
                                [this, response](auto ec) {
                                    std::cout << "[CryptoClient#" << instance_id_ << "] Private WS handshake completed\n";
                                    if (ec) {
                                        std::cerr << "[CryptoClient#" << instance_id_ << "] Private WS handshake failed: "
                                                  << ec.message() << ", HTTP Status: " << response->result_int()
                                                  << ", Reason: " << response->reason() << '\n';
                                        handle_error("Private WS", ec, response->result_int());
                                        return;
                                    }
                                    private_connected_ = true;
                                    std::cout << "[CryptoClient#" << instance_id_ << "] Private WS connected\n";
                                    private_timer_->expires_after(CONNECT_DELAY);
                                    private_timer_->async_wait([this](auto ec) {
                                        std::cout << "[CryptoClient#" << instance_id_ << "] Private subscription timer fired\n";
                                        if (!ec && running_) {
                                            authenticate();
                                            do_private_read();
                                        }
                                    });
                                });
                        });
                });
        });
}

/* ---------------------- authentication & subs ----------------------- */
void CryptoClient::authenticate() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering authenticate\n";
    long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
    std::string method = "public/auth";
    std::string id = "1";

    std::ostringstream prehash;
    prehash << method << id << api_key_ << ts;
    unsigned char digest[32];
    unsigned int digest_len;
    HMAC(EVP_sha256(),
         api_secret_.data(), api_secret_.size(),
         reinterpret_cast<const unsigned char*>(prehash.str().data()), prehash.str().size(),
         digest, &digest_len);

    std::ostringstream sig;
    sig << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_len; ++i) sig << std::setw(2) << static_cast<int>(digest[i]);

    send_private_msg({
        {"id", id},
        {"method", method},
        {"api_key", api_key_},
        {"sig", sig.str()},
        {"nonce", ts}
    });
}

void CryptoClient::subscribe_public() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering subscribe_public\n";
    send_public_msg({
        {"id", "11"},
        {"method", "subscribe"},
        {"params", {{"channels", {"orderbook." + symbol_ + ".20"}}}}
    });
}

void CryptoClient::subscribe_private() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering subscribe_private\n";
    send_private_msg({
        {"id", "12"},
        {"method", "subscribe"},
        {"params", {{"channels", {
            "trade." + symbol_,
            "order." + symbol_,
            "account"
        }}}}
    });
}

/* ---------------------------- send helpers -------------------------- */
void CryptoClient::send_public_msg(json&& j) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering send_public_msg\n";
    if (!public_q_.push(std::move(j))) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Public queue full\n";
        return;
    }
    if (public_ioc_) {
        net::post(*public_ioc_, [this] { write_next_public(); });
    }
}

void CryptoClient::send_private_msg(json&& j) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering send_private_msg\n";
    if (!private_q_.push(std::move(j))) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Private queue full\n";
        return;
    }
    if (private_ioc_) {
        net::post(*private_ioc_, [this] { write_next_private(); });
    }
}

void CryptoClient::write_next_public() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering write_next_public\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!public_connected_ || public_writing_ || !public_ws_) return;
    json j;
    if (public_q_.pop(j)) {
        public_writing_ = true;
        auto dump = j.dump();
        public_ws_->async_write(net::buffer(dump),
            [this](auto ec, std::size_t) {
                std::cout << "[CryptoClient#" << instance_id_ << "] Public write completed\n";
                std::lock_guard<std::mutex> lock(ws_mutex_);
                public_writing_ = false;
                if (ec) handle_error("Public write", ec);
                else if (!public_q_.empty()) write_next_public();
            });
    }
}

void CryptoClient::write_next_private() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering write_next_private\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!private_connected_ || private_writing_ || !private_ws_) return;
    json j;
    if (private_q_.pop(j)) {
        private_writing_ = true;
        auto dump = j.dump();
        private_ws_->async_write(net::buffer(dump),
            [this](auto ec, std::size_t) {
                std::cout << "[CryptoClient#" << instance_id_ << "] Private write completed\n";
                std::lock_guard<std::mutex> lock(ws_mutex_);
                private_writing_ = false;
                if (ec) handle_error("Private write", ec);
                else if (!private_q_.empty()) write_next_private();
            });
    }
}

/* --------------------------- read loops ----------------------------- */
void CryptoClient::do_public_read() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering do_public_read\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!public_ws_ || !public_connected_) return;
    public_ws_->async_read(public_buffer_,
        [this](auto ec, std::size_t) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Public read completed\n";
            if (ec) return handle_error("Public read", ec);
            json j = json::parse(beast::buffers_to_string(public_buffer_.data()), nullptr, false);
            public_buffer_.consume(public_buffer_.size());
            if (!j.is_discarded()) handle_public_msg(j);
            if (running_) do_public_read();
        });
}

void CryptoClient::do_private_read() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering do_private_read\n";
    std::lock_guard<std::mutex> lock(ws_mutex_);
    if (!private_ws_ || !private_connected_) return;
    private_ws_->async_read(private_buffer_,
        [this](auto ec, std::size_t) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Private read completed\n";
            if (ec) return handle_error("Private read", ec);
            json j = json::parse(beast::buffers_to_string(private_buffer_.data()), nullptr, false);
            private_buffer_.consume(private_buffer_.size());
            if (!j.is_discarded()) handle_private_msg(j);
            if (running_) do_private_read();
        });
}

/* ---------------------- message dispatchers ------------------------- */
void CryptoClient::handle_public_msg(const json& j) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering handle_public_msg\n";
    if (!j.contains("method") || j["method"] != "subscribe") return;
    const auto& data = j["result"]["data"];
    if (orderbook_cb_) orderbook_cb_(data);
}

void CryptoClient::handle_private_msg(const json& j) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering handle_private_msg\n";
    if (j.contains("method") && j["method"] == "public/auth") {
        if (j.contains("code") && j["code"] == 0) subscribe_private();
        else std::cerr << "[CryptoClient#" << instance_id_ << "] Auth failed: " << j.dump() << '\n';
        return;
    }
    if (!j.contains("method")) return;
    const std::string m = j["method"];
    const auto& data = j["result"]["data"];
    if (m.rfind("trade", 0) == 0 && trade_cb_) trade_cb_(data);
    else if (m.rfind("order", 0) == 0 && order_cb_) order_cb_(data);
    else if (m == "account" && position_cb_) position_cb_(data);
}

/* ---------------- accessors / callbacks ----------------------------- */
void CryptoClient::set_orderbook_cb(std::function<void(const json&)> cb) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Setting orderbook callback\n";
    orderbook_cb_ = std::move(cb);
}
void CryptoClient::set_trade_cb(std::function<void(const json&)> cb) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Setting trade callback\n";
    trade_cb_ = std::move(cb);
}
void CryptoClient::set_position_cb(std::function<void(const json&)> cb) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Setting position callback\n";
    position_cb_ = std::move(cb);
}
void CryptoClient::set_order_cb(std::function<void(const json&)> cb) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Setting order callback\n";
    order_cb_ = std::move(cb);
}

/* ---------------- rest of trading helpers --------------------------- */
void CryptoClient::place_order(const std::string& side, double price, double size,
                               const std::string& client_oid, bool is_hedge, const std::string& type) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering place_order\n";
    send_private_msg({
        {"id", "20"},
        {"method", "private/create-order"},
        {"params", {
            {"instrument_name", is_hedge ? hedge_symbol_ : symbol_},
            {"side", side},
            {"type", type},
            {"price", price},
            {"quantity", size},
            {"client_oid", client_oid}
        }}
    });
}

void CryptoClient::cancel_order(const std::string& order_id) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering cancel_order\n";
    send_private_msg({ {"id", "21"}, {"method", "private/cancel-order"}, {"params", {{"order_id", order_id}}} });
}

void CryptoClient::cancel_all_orders() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering cancel_all_orders\n";
    send_private_msg({ {"id", "22"}, {"method", "private/cancel-all-orders"}, {"params", {{"instrument_name", symbol_}}} });
}

void CryptoClient::get_position() {
    std::cout << "[CryptoClient#" << instance_id_ << "] Entering get_position\n";
    send_private_msg({ {"id", "23"}, {"method", "private/get-positions"}, {"params", {{"instrument_name", symbol_}}} });
}

/* --------------------------- error handler -------------------------- */
void CryptoClient::handle_error(const std::string& where, const beast::error_code& ec, int http_status) {
    std::cerr << "[CryptoClient#" << instance_id_ << "] " << where << " : " << ec.message() << '\n';
    if (ec == boost::asio::error::connection_refused || ec == boost::asio::ssl::error::stream_truncated || ec) {
        // Retry connection with exponential backoff
        auto delay = std::chrono::milliseconds(http_status == 429 ? 60000 : 30000 * (1 + retry_count_++));
        std::cerr << "[CryptoClient#" << instance_id_ << "] Retrying " << where << " after " << delay.count() << "ms\n";
        if (where.find("Public") != std::string::npos) {
            if (public_timer_) {
                public_timer_->expires_after(delay);
                public_timer_->async_wait([this](auto ec) {
                    std::cout << "[CryptoClient#" << instance_id_ << "] Public retry timer fired\n";
                    if (!ec && running_) do_public_connect();
                });
            }
        } else {
            if (private_timer_) {
                private_timer_->expires_after(delay);
                private_timer_->async_wait([this](auto ec) {
                    std::cout << "[CryptoClient#" << instance_id_ << "] Private retry timer fired\n";
                    if (!ec && running_) do_private_connect();
                });
            }
        }
    } else {
        retry_count_ = 0; // Reset retry count on non-retryable errors
    }
}

} // namespace RLTrader
