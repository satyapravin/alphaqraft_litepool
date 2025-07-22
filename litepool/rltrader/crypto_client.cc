#include "crypto_client.h"
#include <boost/asio/connect.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace RLTrader {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace ws = boost::beast::websocket;
using tcp = boost::asio::ip::tcp;
using ssl_socket = boost::asio::ssl::stream<beast::tcp_stream>;
using websocket_stream = ws::stream<ssl_socket>;

// Constants
constexpr std::chrono::seconds WS_PING_INTERVAL{1};
constexpr std::chrono::seconds APP_PING_INTERVAL{30};
constexpr std::chrono::seconds RECONNECT_DELAY{1};
constexpr std::chrono::seconds READ_TIMEOUT{60};

static constexpr const char* CR_PUBLIC_HOST = "stream.crypto.com";
static constexpr const char* CR_PRIVATE_HOST = "stream.crypto.com";
static constexpr const char* CR_PUBLIC_PATH = "/exchange/v1/market";
static constexpr const char* CR_PRIVATE_PATH = "/exchange/v1/user";
static constexpr const char* CR_SSL_PORT = "443";
static constexpr const char* USER_AGENT = "AlphaQraft_Trading";

CryptoClient::CryptoClient(std::string api_key, std::string api_secret,
                         std::string symbol, std::string hedge_symbol)
    : api_key_(std::move(api_key)),
      api_secret_(std::move(api_secret)),
      symbol_(std::move(symbol)),
      hedge_symbol_(hedge_symbol.empty() ? symbol_ : std::move(hedge_symbol)),
      instance_id_(gen_id()) {
    std::cout << "[CryptoClient#" << instance_id_ << "] Created\n";
}

CryptoClient::~CryptoClient() {
    stop();
}

void CryptoClient::start() {
    if (running_.exchange(true)) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Already running\n";
        return;
    }

    try {
        public_ioc_ = std::make_unique<boost::asio::io_context>();
        private_ioc_ = std::make_unique<boost::asio::io_context>();
        
        public_write_strand_ = std::make_unique<boost::asio::strand<boost::asio::io_context::executor_type>>(
            public_ioc_->get_executor());
        private_write_strand_ = std::make_unique<boost::asio::strand<boost::asio::io_context::executor_type>>(
            private_ioc_->get_executor());

        ssl_ctx_ = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12_client);
        ssl_ctx_->set_default_verify_paths();
        ssl_ctx_->set_verify_mode(boost::asio::ssl::verify_peer);
        SSL_CTX_set_options(ssl_ctx_->native_handle(), SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);
        SSL_CTX_set_cipher_list(ssl_ctx_->native_handle(), "HIGH:!aNULL:!kRSA:!PSK:!SRP:!MD5:!RC4");

        public_work_ = std::make_unique<boost::asio::executor_work_guard<
            boost::asio::io_context::executor_type>>(public_ioc_->get_executor());
        private_work_ = std::make_unique<boost::asio::executor_work_guard<
            boost::asio::io_context::executor_type>>(private_ioc_->get_executor());

        public_thread_ = std::thread([this] {
            public_ioc_->run();
        });

        private_thread_ = std::thread([this] {
            private_ioc_->run();
        });

        setup_public_ws();
        setup_private_ws();

    } catch (const std::exception& e) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Start failed: " << e.what() << "\n";
        stop();
        throw;
    }
}

void CryptoClient::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    // Cancel timers
    if (public_ping_timer_) public_ping_timer_->cancel();
    if (private_ping_timer_) private_ping_timer_->cancel();
    if (public_heartbeat_timer_) public_heartbeat_timer_->cancel();
    if (private_heartbeat_timer_) private_heartbeat_timer_->cancel();

    // Close WebSocket connections
    {
        std::lock_guard lock(public_mutex_);
        if (public_ws_ && public_ws_->is_open()) {
            public_ws_->async_close(ws::close_code::normal,
                boost::asio::bind_executor(*public_write_strand_,
                    [this](beast::error_code ec) {
                        if (ec) {
                            std::cerr << "[CryptoClient#" << instance_id_
                                      << "] Public WebSocket close error: " << ec.message() << "\n";
                        }
                    }));
        }
    }
    {
        std::lock_guard lock(private_mutex_);
        if (private_ws_ && private_ws_->is_open()) {
            private_ws_->async_close(ws::close_code::normal,
                boost::asio::bind_executor(*private_write_strand_,
                    [this](beast::error_code ec) {
                        if (ec) {
                            std::cerr << "[CryptoClient#" << instance_id_
                                      << "] Private WebSocket close error: " << ec.message() << "\n";
                        }
                    }));
        }
    }

    // Stop IO contexts
    if (public_ioc_) public_ioc_->stop();
    if (private_ioc_) private_ioc_->stop();

    // Join threads
    if (public_thread_.joinable()) public_thread_.join();
    if (private_thread_.joinable()) private_thread_.join();

    // Reset resources
    {
        std::lock_guard lock(public_mutex_);
        public_ws_.reset();
        public_resolver_.reset();
        public_ping_timer_.reset();
        public_heartbeat_timer_.reset();
        public_work_.reset();
        public_ioc_.reset();
    }
    {
        std::lock_guard lock(private_mutex_);
        private_ws_.reset();
        private_resolver_.reset();
        private_ping_timer_.reset();
        private_heartbeat_timer_.reset();
        private_work_.reset();
        private_ioc_.reset();
    }

    ssl_ctx_.reset();
}

void CryptoClient::setup_public_ws() {
    std::lock_guard lock(public_mutex_);
    public_ws_ = std::make_unique<websocket_stream>(*public_ioc_, *ssl_ctx_);
    public_ping_timer_ = std::make_unique<net::steady_timer>(*public_ioc_);
    public_heartbeat_timer_ = std::make_unique<net::steady_timer>(*public_ioc_);

    public_ws_->control_callback(
        [this](ws::frame_type type, beast::string_view payload) {
            if (type == ws::frame_type::pong) {
                std::cout << "[CryptoClient#" << instance_id_
                          << "] Received WebSocket pong: " << payload << "\n";
            }
        });

    public_ws_->set_option(ws::stream_base::decorator(
        [](ws::request_type& req) {
            req.set(beast::http::field::host, CR_PUBLIC_HOST);
            req.set(beast::http::field::user_agent, USER_AGENT);
            req.set(beast::http::field::accept, "*/*");
            req.set(beast::http::field::connection, "upgrade");
            req.set(beast::http::field::upgrade, "websocket");
            req.set(beast::http::field::sec_websocket_version, "13");
        }));

    do_public_connect();
}

void CryptoClient::do_public_connect() {
    if (!running_) return;

    std::cout << "[CryptoClient#" << instance_id_ << "] Connecting to public stream...\n";

    public_resolver_ = std::make_unique<tcp::resolver>(*public_ioc_);

    public_resolver_->async_resolve(CR_PUBLIC_HOST, CR_SSL_PORT,
        boost::asio::bind_executor(*public_write_strand_,
            [this](const boost::system::error_code& ec, tcp::resolver::results_type results) {
                if (ec) {
                    return handle_public_error("resolve", ec);
                }

                auto stream = std::make_shared<beast::tcp_stream>(*public_ioc_);

                stream->async_connect(results,
                    boost::asio::bind_executor(*public_write_strand_,
                        [this, stream](const boost::system::error_code& ec, tcp::resolver::results_type::endpoint_type) {
                            if (ec) {
                                return handle_public_error("connect", ec);
                            }

                            auto ssl_stream = std::make_shared<net::ssl::stream<beast::tcp_stream>>(
                                std::move(*stream), *ssl_ctx_);

                            if (!SSL_set_tlsext_host_name(ssl_stream->native_handle(), CR_PUBLIC_HOST)) {
                                beast::error_code ec{static_cast<int>(::ERR_get_error()),
                                                     net::error::get_ssl_category()};
                                return handle_public_error("SNI", ec);
                            }

                            ssl_stream->async_handshake(net::ssl::stream_base::client,
                                boost::asio::bind_executor(*public_write_strand_,
                                    [this, ssl_stream](const boost::system::error_code& ec) {
                                        if (ec) {
                                            return handle_public_error("SSL handshake", ec);
                                        }

                                        std::lock_guard lock(public_mutex_);
                                        public_ws_ = std::make_unique<websocket_stream>(std::move(*ssl_stream));
                                        public_ws_->binary(true);

                                        public_ws_->async_handshake(CR_PUBLIC_HOST, CR_PUBLIC_PATH,
                                            boost::asio::bind_executor(*public_write_strand_,
                                                [this](const boost::system::error_code& ec) {
                                                    if (ec) {
                                                        return handle_public_error("WS handshake", ec);
                                                    }

                                                    std::cout << "[CryptoClient#" << instance_id_
                                                              << "] Public WebSocket connected\n";
                                                    public_connected_ = true;
                                                    start_public_ping();
                                                    subscribe_public();
                                                    do_public_read();
                                                }));
                                    }));
                        }));
            }));
}

void CryptoClient::start_public_ping() {
    if (!running_) return;

    public_ping_timer_->expires_after(WS_PING_INTERVAL);
    public_ping_timer_->async_wait(
        boost::asio::bind_executor(*public_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (ec || !running_) {
                    return;
                }

                std::lock_guard lock(public_mutex_);
                if (public_ws_ && public_ws_->is_open()) {
                    public_ws_->async_ping("",
                        boost::asio::bind_executor(*public_write_strand_,
                            [this](const boost::system::error_code& ec) {
                                if (ec) {
                                    handle_public_error("ping", ec);
                                } else {
                                    start_public_ping();
                                }
                            }));
                }
            }));

    public_heartbeat_timer_->expires_after(APP_PING_INTERVAL);
    public_heartbeat_timer_->async_wait(
        boost::asio::bind_executor(*public_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (ec || !running_) {
                    return;
                }

                json ping_msg = {
                    {"id", std::to_string(std::time(nullptr))},
                    {"method", "public/ping"}
                };
                send_public_msg(std::move(ping_msg));
                public_heartbeat_timer_->expires_after(APP_PING_INTERVAL);
                public_heartbeat_timer_->async_wait(
                    boost::asio::bind_executor(*public_write_strand_,
                        [this](const boost::system::error_code& ec) {
                            if (ec || !running_) return;
                            start_public_ping();
                        }));
            }));
}

void CryptoClient::do_public_read() {
    if (!running_) return;

    std::lock_guard lock(public_mutex_);
    if (!public_ws_ || !public_ws_->is_open()) {
        return;
    }

    public_buffer_.consume(public_buffer_.size());

    auto timeout = std::make_shared<net::steady_timer>(*public_ioc_);
    timeout->expires_after(READ_TIMEOUT);
    timeout->async_wait(
        boost::asio::bind_executor(*public_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (!ec && running_) {
                    std::lock_guard lock(public_mutex_);
                    if (public_ws_ && public_ws_->is_open()) {
                        public_ws_->async_close(ws::close_code::normal,
                            boost::asio::bind_executor(*public_write_strand_,
                                [this](const boost::system::error_code& ec) {
                                    if (ec) {
                                        std::cerr << "[CryptoClient#" << instance_id_
                                                  << "] Timeout close error: " << ec.message() << "\n";
                                    }
                                }));
                    }
                }
            }));

    public_ws_->async_read(public_buffer_,
        boost::asio::bind_executor(*public_write_strand_,
            [this, timeout](const boost::system::error_code& ec, std::size_t bytes_transferred) {
                timeout->cancel();

                if (ec == ws::error::closed || ec == net::error::operation_aborted) {
                    std::cout << "[CryptoClient#" << instance_id_ << "] Public connection closed\n";
                    return;
                }

                if (ec) {
                    return handle_public_error("read", ec);
                }

                try {
                    std::string data = beast::buffers_to_string(public_buffer_.data());
                    public_buffer_.consume(bytes_transferred);

                    json j = json::parse(data, nullptr, false);
                    if (j.is_discarded()) {
                        std::cerr << "[CryptoClient#" << instance_id_
                                  << "] Failed to parse message: " << data << "\n";
                    } else {
                        handle_public_msg(j);
                    }

                    if (running_) {
                        do_public_read();
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[CryptoClient#" << instance_id_
                              << "] Message processing error: " << e.what() << "\n";
                    if (running_) {
                        do_public_read();
                    }
                }
            }));
}

void CryptoClient::handle_public_error(const std::string& where, beast::error_code ec) {
    if (!running_) return;

    std::cerr << "[CryptoClient#" << instance_id_ << "] Public error in "
              << where << ": " << ec.message() << "\n";

    public_connected_ = false;

    if (ec == net::error::operation_aborted) {
        return;
    }

    auto timer = std::make_unique<net::steady_timer>(*public_ioc_);
    timer->expires_after(RECONNECT_DELAY);
    timer->async_wait(
        boost::asio::bind_executor(*public_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (!ec && running_) {
                    do_public_connect();
                }
            }));
}

void CryptoClient::send_public_msg(json&& j) {
    if (!running_ || !public_connected_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Public stream not connected, dropping message\n";
        return;
    }

    std::lock_guard lock(public_mutex_);
    if (!public_ws_ || !public_ws_->is_open()) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Public WebSocket not open, dropping message\n";
        return;
    }

    boost::asio::post(*public_write_strand_,
        [this, msg = j.dump()]() {
            std::cout << "[CryptoClient#" << instance_id_ << "] Sending public message: " << msg << "\n";
            public_ws_->async_write(net::buffer(msg),
                boost::asio::bind_executor(*public_write_strand_,
                    [this](const boost::system::error_code& ec, std::size_t) {
                        if (ec) {
                            handle_public_error("write", ec);
                        }
                    }));
        });
}

void CryptoClient::subscribe_public() {
    json msg = {
        {"id", "1"},
        {"method", "subscribe"},
        {"params", {
            {"channels", {"book." + symbol_ + ".50"}},
            {"book_subscription_type", "SNAPSHOT"}
        }}
    };
    send_public_msg(std::move(msg));
}

void CryptoClient::handle_public_msg(const json& j) {
    if (!j.contains("method")) {
        std::cerr << "[CryptoClient#" << instance_id_
                  << "] Message missing method: " << j.dump() << "\n";
        return;
    }

    const std::string method = j["method"];
    if (method == "public/heartbeat") {
        if (j.contains("id")) {
            json response = {
                {"id", j["id"]},
                {"method", "public/respond-heartbeat"}
            };
            send_public_msg(std::move(response));
        }
    } else if (method == "subscribe" && j.contains("result")) {
        if (orderbook_cb_) {
            orderbook_cb_(j["result"]);
        }
    }
}

void CryptoClient::setup_private_ws() {
    std::lock_guard lock(private_mutex_);
    private_ws_ = std::make_unique<websocket_stream>(*private_ioc_, *ssl_ctx_);
    private_ping_timer_ = std::make_unique<net::steady_timer>(*private_ioc_);
    private_heartbeat_timer_ = std::make_unique<net::steady_timer>(*private_ioc_);

    private_ws_->control_callback(
        [this](ws::frame_type type, beast::string_view payload) {
            if (type == ws::frame_type::pong) {
                std::cout << "[CryptoClient#" << instance_id_
                          << "] Received private pong: " << payload << "\n";
            }
        });

    private_ws_->set_option(ws::stream_base::decorator(
        [](ws::request_type& req) {
            req.set(beast::http::field::host, CR_PRIVATE_HOST);
            req.set(beast::http::field::user_agent, USER_AGENT);
            req.set(beast::http::field::accept, "*/*");
            req.set(beast::http::field::connection, "upgrade");
            req.set(beast::http::field::upgrade, "websocket");
            req.set(beast::http::field::sec_websocket_version, "13");
        }));

    do_private_connect();
}

void CryptoClient::do_private_connect() {
    if (!running_) return;

    std::cout << "[CryptoClient#" << instance_id_ << "] Connecting to private stream...\n";

    private_resolver_ = std::make_unique<tcp::resolver>(*private_ioc_);

    private_resolver_->async_resolve(CR_PRIVATE_HOST, CR_SSL_PORT,
        boost::asio::bind_executor(*private_write_strand_,
            [this](const boost::system::error_code& ec, tcp::resolver::results_type results) {
                if (ec) {
                    return handle_private_error("resolve", ec);
                }

                auto stream = std::make_shared<beast::tcp_stream>(*private_ioc_);

                stream->async_connect(results,
                    boost::asio::bind_executor(*private_write_strand_,
                        [this, stream](const boost::system::error_code& ec, tcp::resolver::results_type::endpoint_type) {
                            if (ec) {
                                return handle_private_error("connect", ec);
                            }

                            auto ssl_stream = std::make_shared<net::ssl::stream<beast::tcp_stream>>(
                                std::move(*stream), *ssl_ctx_);

                            if (!SSL_set_tlsext_host_name(ssl_stream->native_handle(), CR_PRIVATE_HOST)) {
                                beast::error_code ec{static_cast<int>(::ERR_get_error()),
                                                     net::error::get_ssl_category()};
                                return handle_private_error("SNI", ec);
                            }

                            ssl_stream->async_handshake(net::ssl::stream_base::client,
                                boost::asio::bind_executor(*private_write_strand_,
                                    [this, ssl_stream](const boost::system::error_code& ec) {
                                        if (ec) {
                                            return handle_private_error("SSL handshake", ec);
                                        }

                                        std::lock_guard lock(private_mutex_);
                                        private_ws_ = std::make_unique<websocket_stream>(std::move(*ssl_stream));
                                        private_ws_->binary(true);

                                        private_ws_->async_handshake(CR_PRIVATE_HOST, CR_PRIVATE_PATH,
                                            boost::asio::bind_executor(*private_write_strand_,
                                                [this](const boost::system::error_code& ec) {
                                                    if (ec) {
                                                        return handle_private_error("WS handshake", ec);
                                                    }

                                                    std::cout << "[CryptoClient#" << instance_id_
                                                              << "] Private WebSocket connected\n";
                                                    private_connected_ = true;
                                                    start_private_ping();
                                                    authenticate();
                                                }));
                                    }));
                        }));
            }));
}

void CryptoClient::start_private_ping() {
    if (!running_) return;

    private_ping_timer_->expires_after(WS_PING_INTERVAL);
    private_ping_timer_->async_wait(
        boost::asio::bind_executor(*private_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (ec || !running_) {
                    return;
                }

                std::lock_guard lock(private_mutex_);
                if (private_ws_ && private_ws_->is_open()) {
                    private_ws_->async_ping("",
                        boost::asio::bind_executor(*private_write_strand_,
                            [this](const boost::system::error_code& ec) {
                                if (ec) {
                                    handle_private_error("ping", ec);
                                } else {
                                    start_private_ping();
                                }
                            }));
                }
            }));

    private_heartbeat_timer_->expires_after(APP_PING_INTERVAL);
    private_heartbeat_timer_->async_wait(
        boost::asio::bind_executor(*private_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (ec || !running_) {
                    return;
                }

                json ping_msg = {
                    {"id", std::to_string(std::time(nullptr))},
                    {"method", "public/ping"}
                };
                send_private_msg(std::move(ping_msg));
                private_heartbeat_timer_->expires_after(APP_PING_INTERVAL);
                private_heartbeat_timer_->async_wait(
                    boost::asio::bind_executor(*private_write_strand_,
                        [this](const boost::system::error_code& ec) {
                            if (ec || !running_) return;
                            start_private_ping();
                        }));
            }));
}

void CryptoClient::authenticate() {
    const long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    const std::string payload = build_payload("public/auth", "1", api_key_, json::object(), ts);

    unsigned char digest[32];
    unsigned int dlen{};
    HMAC(EVP_sha256(), api_secret_.data(), api_secret_.size(),
         reinterpret_cast<const unsigned char*>(payload.data()),
         payload.size(), digest, &dlen);

    std::ostringstream sig;
    sig << std::hex << std::setfill('0');
    for (unsigned i = 0; i < dlen; ++i) {
        sig << std::setw(2) << static_cast<int>(digest[i]);
    }

    json msg = {
        {"id", "1"},
        {"method", "public/auth"},
        {"api_key", api_key_},
        {"sig", sig.str()},
        {"nonce", ts}
    };

    send_private_msg(std::move(msg));
}

void CryptoClient::send_private_msg(json&& j) {
    if (!running_ || !private_connected_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Private stream not connected, dropping message\n";
        return;
    }

    std::lock_guard lock(private_mutex_);
    if (!private_ws_ || !private_ws_->is_open()) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Private WebSocket not open, dropping message\n";
        return;
    }

    boost::asio::post(*private_write_strand_,
        [this, msg = j.dump()]() {
            std::cout << "[CryptoClient#" << instance_id_ << "] Sending private message: " << msg << "\n";
            private_ws_->async_write(net::buffer(msg),
                boost::asio::bind_executor(*private_write_strand_,
                    [this](const boost::system::error_code& ec, std::size_t) {
                        if (ec) {
                            handle_private_error("write", ec);
                        }
                    }));
        });
}

void CryptoClient::handle_private_error(const std::string& where, beast::error_code ec) {
    if (!running_) return;

    std::cerr << "[CryptoClient#" << instance_id_ << "] Private error in "
              << where << ": " << ec.message() << "\n";

    private_connected_ = false;

    if (ec == net::error::operation_aborted) {
        return;
    }

    auto timer = std::make_unique<net::steady_timer>(*private_ioc_);
    timer->expires_after(RECONNECT_DELAY);
    timer->async_wait(
        boost::asio::bind_executor(*private_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (!ec && running_) {
                    do_private_connect();
                }
            }));
}

void CryptoClient::do_private_read() {
    if (!running_) return;

    std::lock_guard lock(private_mutex_);
    if (!private_ws_ || !private_ws_->is_open()) {
        return;
    }

    private_buffer_.consume(private_buffer_.size());

    auto timeout = std::make_shared<net::steady_timer>(*private_ioc_);
    timeout->expires_after(READ_TIMEOUT);
    timeout->async_wait(
        boost::asio::bind_executor(*private_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (!ec && running_) {
                    std::lock_guard lock(private_mutex_);
                    if (private_ws_ && private_ws_->is_open()) {
                        private_ws_->async_close(ws::close_code::normal,
                            boost::asio::bind_executor(*private_write_strand_,
                                [this](const boost::system::error_code& ec) {
                                    if (ec) {
                                        std::cerr << "[CryptoClient#" << instance_id_
                                                  << "] Private timeout close error: " << ec.message() << "\n";
                                    }
                                }));
                    }
                }
            }));

    private_ws_->async_read(private_buffer_,
        boost::asio::bind_executor(*private_write_strand_,
            [this, timeout](const boost::system::error_code& ec, std::size_t bytes_transferred) {
                timeout->cancel();

                if (ec == ws::error::closed || ec == net::error::operation_aborted) {
                    std::cout << "[CryptoClient#" << instance_id_ << "] Private connection closed\n";
                    return;
                }

                if (ec) {
                    return handle_private_error("read", ec);
                }

                try {
                    std::string data = beast::buffers_to_string(private_buffer_.data());
                    private_buffer_.consume(bytes_transferred);

                    json j = json::parse(data, nullptr, false);
                    if (j.is_discarded()) {
                        std::cerr << "[CryptoClient#" << instance_id_
                                  << "] Failed to parse private message: " << data << "\n";
                    } else {
                        handle_private_msg(j);
                    }

                    if (running_) {
                        do_private_read();
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[CryptoClient#" << instance_id_
                              << "] Private message processing error: " << e.what() << "\n";
                    if (running_) {
                        do_private_read();
                    }
                }
            }));
}

void CryptoClient::subscribe_private() {
    private_heartbeat_timer_->expires_after(std::chrono::seconds(2));
    private_heartbeat_timer_->async_wait(
        boost::asio::bind_executor(*private_write_strand_,
            [this](const boost::system::error_code& ec) {
                if (ec || !running_) {
                    return;
                }

                json msg = {
                    {"id", "2"},
                    {"method", "subscribe"},
                    {"params", {
                        {"channels", {"user.order." + symbol_, "user.position." + symbol_}}
                    }}
                };
                send_private_msg(std::move(msg));
            }));
}

void CryptoClient::handle_private_msg(const json& j) {
    if (!j.contains("method")) {
        std::cerr << "[CryptoClient#" << instance_id_
                  << "] Private message missing method: " << j.dump() << "\n";
        return;
    }

    const std::string method = j["method"];
    if (method == "public/heartbeat") {
        if (j.contains("id")) {
            json response = {
                {"id", j["id"]},
                {"method", "public/respond-heartbeat"}
            };
            send_private_msg(std::move(response));
        }
    } else if (method == "public/auth" && j.contains("result")) {
        std::cout << "[CryptoClient#" << instance_id_ << "] Authentication successful\n";
        subscribe_private();
        do_private_read();
    } else if (method == "subscribe" && j.contains("result")) {
        if (j["result"].contains("channel")) {
            const std::string channel = j["result"]["channel"];
            if (channel.find("user.order") != std::string::npos && order_cb_) {
                order_cb_(j["result"]);
            } else if (channel.find("user.position") != std::string::npos && position_cb_) {
                position_cb_(j["result"]);
            } else if (channel.find("user.trade") != std::string::npos && trade_cb_) {
                trade_cb_(j["result"]);
            }
        }
    }
}

void CryptoClient::set_orderbook_cb(std::function<void(const json&)> cb) {
    orderbook_cb_ = std::move(cb);
}

void CryptoClient::set_trade_cb(std::function<void(const json&)> cb) {
    trade_cb_ = std::move(cb);
}

void CryptoClient::set_position_cb(std::function<void(const json&)> cb) {
    position_cb_ = std::move(cb);
}

void CryptoClient::set_order_cb(std::function<void(const json&)> cb) {
    order_cb_ = std::move(cb);
}

void CryptoClient::place_order(const std::string& side, double price, double size,
                              const std::string& client_oid, bool hedge,
                              const std::string& type) {
    json msg = {
        {"id", "2"},
        {"method", "private/create-order"},
        {"params", {
            {"instrument_name", hedge ? hedge_symbol_ : symbol_},
            {"side", side},
            {"type", type},
            {"price", price},
            {"quantity", size},
            {"client_oid", client_oid}
        }}
    };
    send_private_msg(std::move(msg));
}

void CryptoClient::cancel_order(const std::string& order_id) {
    json msg = {
        {"id", "3"},
        {"method", "private/cancel-order"},
        {"params", {{"order_id", order_id}}}
    };
    send_private_msg(std::move(msg));
}

void CryptoClient::cancel_all_orders() {
    json msg = {
        {"id", "4"},
        {"method", "private/cancel-all-orders"},
        {"params", {{"instrument_name", symbol_}}}
    };
    send_private_msg(std::move(msg));
}

void CryptoClient::get_position() {
    json msg = {
        {"id", "5"},
        {"method", "private/get-positions"},
        {"params", {{"instrument_name", symbol_}}}
    };
    send_private_msg(std::move(msg));
}

std::string CryptoClient::gen_id() const {
    static std::atomic<int> counter{0};
    return std::to_string(counter++);
}

std::string CryptoClient::build_payload(const std::string& method, const std::string& id,
                                       const std::string& api_key, const json& params, long nonce) const {
    json payload = {
        {"id", id},
        {"method", method},
        {"api_key", api_key},
        {"params", params},
        {"nonce", nonce}
    };
    return payload.dump();
}

} // namespace RLTrader
