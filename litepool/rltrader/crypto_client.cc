#include "crypto_client.h"
#include "crypto_util.h"
#include <boost/asio/connect.hpp>
#include <boost/asio/post.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace RLTrader
{
    namespace net   = boost::asio;
    namespace ssl   = boost::asio::ssl;
    namespace beast = boost::beast;
    namespace ws    = beast::websocket;

    using tcp  = net::ip::tcp;

/* ------------------------------------------------------------------ */
/*  Compile-time constants                                            */
/* ------------------------------------------------------------------ */
constexpr const char* CR_PUBLIC_HOST  = "uat-stream.3ona.co";
constexpr const char* CR_PRIVATE_HOST = "uat-stream.3ona.co";
// constexpr const char* CR_PUBLIC_HOST  = "stream.crypto.com";
// constexpr const char* CR_PRIVATE_HOST = "stream.crypto.com";

constexpr const char* CR_PUBLIC_PATH  = "/exchange/v1/market";
constexpr const char* CR_PRIVATE_PATH = "/exchange/v1/user";
constexpr const char* CR_SSL_PORT     = "443";

constexpr const char* USER_AGENT      = "AlphaQraft_Trading";

static   constexpr std::chrono::milliseconds CONNECT_DELAY {1000};
static   constexpr std::chrono::seconds      CONNECT_TIMEOUT{10};
static   constexpr std::chrono::seconds      READ_TIMEOUT{30};

/* ------------------------------------------------------------------ */
/*  Constructor / destructor                                          */
/* ------------------------------------------------------------------ */
CryptoClient::CryptoClient(std::string api_key,
                           std::string api_secret,
                           std::string symbol,
                           std::string hedge_symbol)
    : api_key_(std::move(api_key)),
      api_secret_(std::move(api_secret)),
      symbol_(std::move(symbol)),
      hedge_symbol_(hedge_symbol.empty() ? symbol_ : std::move(hedge_symbol)),
      instance_id_(gen_id())
{
    std::cout << "[CryptoClient#" << instance_id_ << "] constructed\n";
}

CryptoClient::~CryptoClient() { stop(); }

/* ------------------------------------------------------------------ */
/*  Public life-cycle                                                 */
/* ------------------------------------------------------------------ */
void CryptoClient::start()
{
    if (running_.exchange(true)) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] already running\n";
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    /* fresh IO contexts & SSL -------------------------------------- */
    {
        std::lock_guard lp(public_mutex_);
        std::lock_guard lq(private_mutex_);

        public_ioc_  = std::make_unique<net::io_context>();
        private_ioc_ = std::make_unique<net::io_context>();

        ssl_ctx_ = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
        ssl_ctx_->set_default_verify_paths();
        ssl_ctx_->set_verify_mode(ssl::verify_peer);

        public_resolver_  = std::make_unique<tcp::resolver>(*public_ioc_);
        private_resolver_ = std::make_unique<tcp::resolver>(*private_ioc_);

        public_work_  = std::make_unique<net::executor_work_guard<net::io_context::executor_type>>(public_ioc_->get_executor());
        private_work_ = std::make_unique<net::executor_work_guard<net::io_context::executor_type>>(private_ioc_->get_executor());
    }

    /* threads ------------------------------------------------------- */
    public_thread_  = std::make_unique<std::thread>([this] {
        std::cout << "[CryptoClient#" << instance_id_ << "] public IO thread running\n";
        public_ioc_->run();
    });
    private_thread_ = std::make_unique<std::thread>([this] {
        std::cout << "[CryptoClient#" << instance_id_ << "] private IO thread running\n";
        private_ioc_->run();
    });

    setup_connections();
}

void CryptoClient::stop()
{
    if (!running_.exchange(false)) return;

    boost::system::error_code ec;

    // Close WebSocket streams first
    {
        std::lock_guard lp(public_mutex_);
        if (public_ws_) {
            public_ws_->async_close(ws::close_code::normal, 
                [this](auto ec) {
                    if (ec) handle_error("public close", ec);
                });
        }
    }
    {
        std::lock_guard lq(private_mutex_);
        if (private_ws_) {
            private_ws_->async_close(ws::close_code::normal, 
                [this](auto ec) {
                    if (ec) handle_error("private close", ec);
                });
        }
    }

    // Cancel timers after closing streams
    {
        std::lock_guard lp(public_mutex_);
        if (public_timer_) public_timer_->cancel(ec);
        if (public_work_) public_work_.reset();
        if (public_ioc_) public_ioc_->stop();
    }
    {
        std::lock_guard lq(private_mutex_);
        if (private_timer_) private_timer_->cancel(ec);
        if (private_work_) private_work_.reset();
        if (private_ioc_) private_ioc_->stop();
    }

    // Join threads
    if (public_thread_ && public_thread_->joinable()) public_thread_->join();
    if (private_thread_ && private_thread_->joinable()) private_thread_->join();

    // Clear resources
    {
        std::lock_guard lp(public_mutex_);
        std::lock_guard lq(private_mutex_);
        public_ws_.reset();
        private_ws_.reset();
        public_resolver_.reset();
        private_resolver_.reset();
        ssl_ctx_.reset();
        public_ioc_.reset();
        private_ioc_.reset();
        public_thread_.reset();
        private_thread_.reset();
    }

    public_connected_ = false;
    private_connected_ = false;
    public_writing_ = false;
    private_writing_ = false;
    retry_count_ = 0;

    std::cout << "[CryptoClient#" << instance_id_ << "] Stopped cleanly\n";
}

/* ------------------------------------------------------------------ */
/*  Connection bootstrap                                              */
/* ------------------------------------------------------------------ */
void CryptoClient::setup_connections()
{
    setup_public_ws();
    setup_private_ws();
}

/* ---------------- public stream ----------------------------------- */
void CryptoClient::setup_public_ws()
{
    std::lock_guard lp(public_mutex_);
    public_ws_ = std::make_unique<websocket_stream>(*public_ioc_, *ssl_ctx_);
    public_timer_ = std::make_unique<net::steady_timer>(*public_ioc_);

    public_ws_->set_option(ws::stream_base::decorator(
        [](ws::request_type& req) {
            req.set(beast::http::field::host, CR_PUBLIC_HOST);
            req.set(beast::http::field::user_agent, USER_AGENT);
            req.set(beast::http::field::sec_websocket_protocol, "json");
        }));

    //public_timer_->expires_after(CONNECT_DELAY);
    public_timer_->async_wait([this](auto ec) {
        if (!ec && running_) do_public_connect();
    });
}

void CryptoClient::do_public_connect()
{
    std::cout << "[CryptoClient#" << instance_id_ << "] Starting public connect\n";
    auto timeout = std::make_shared<net::steady_timer>(*public_ioc_);
    timeout->expires_after(CONNECT_TIMEOUT);
    timeout->async_wait([this](auto ec) {
        if (!ec) handle_error("public connect timeout", net::error::timed_out);
    });

    public_resolver_->async_resolve(CR_PUBLIC_HOST, CR_SSL_PORT,
        [this, timeout](auto ec, auto results) {
            if (ec) { timeout->cancel(); return handle_error("public resolve", ec); }

            auto stream = std::make_shared<beast::tcp_stream>(*public_ioc_);
            stream->expires_after(CONNECT_TIMEOUT);

            beast::get_lowest_layer(*stream).async_connect(results,
                [this, stream, timeout](auto ec, auto) {
                    if (ec) { timeout->cancel(); return handle_error("public connect", ec); }

                    auto ssl_stream = std::make_shared<ssl::stream<beast::tcp_stream>>(
                        std::move(*stream), *ssl_ctx_);
                    
                    if (!SSL_set_tlsext_host_name(ssl_stream->native_handle(), CR_PUBLIC_HOST)) {
                        return handle_error("SNI config",
                            beast::error_code(
                                static_cast<int>(::ERR_get_error()),
                                net::error::get_ssl_category()));
                    }

                    ssl_stream->async_handshake(ssl::stream_base::client,
                        [this, ssl_stream, timeout](auto ec) {
                            if (ec) { timeout->cancel(); return handle_error("public SSL", ec); }

                            {
			        SSL_set_mode(ssl_stream->native_handle(), SSL_MODE_AUTO_RETRY);
                                beast::get_lowest_layer(*ssl_stream).socket().set_option( boost::asio::socket_base::keep_alive(true));
                                std::lock_guard lp(public_mutex_);
                                public_ws_ = std::make_unique<websocket_stream>(std::move(*ssl_stream));
                                public_ws_->auto_fragment(false);
                                public_ws_->set_option(ws::stream_base::decorator(
                                    [](ws::request_type& req) {
                                        req.set(beast::http::field::host, CR_PUBLIC_HOST);
                                        req.set(beast::http::field::user_agent, USER_AGENT);
                                        req.set(beast::http::field::sec_websocket_protocol, "json");
                                    }));
                            }

                            public_ws_->async_handshake(CR_PUBLIC_HOST, CR_PUBLIC_PATH,
                                [this, timeout](auto ec) {
                                    timeout->cancel();
                                    if (ec) return handle_error("public handshake", ec);
                                    std::cout << "[CryptoClient#" << instance_id_ << "] Public handshake completed\n";
                                    public_connected_ = true;
                                    subscribe_public();
                                    do_public_read();
                                });
                        });
                });
        });
}

/* ---------------- private stream ---------------------------------- */
void CryptoClient::setup_private_ws()
{
    {
        std::lock_guard lq(private_mutex_);
        private_ws_   = std::make_unique<websocket_stream>(*private_ioc_, *ssl_ctx_);
        private_timer_= std::make_unique<net::steady_timer>(*private_ioc_);
        private_ws_->set_option(ws::stream_base::decorator(
            [](ws::request_type& req){
                req.set(beast::http::field::user_agent, USER_AGENT);
                req.set(beast::http::field::host, CR_PRIVATE_HOST);
            }));
    }

    //private_timer_->expires_after(CONNECT_DELAY);
    private_timer_->async_wait([this](auto ec){
        if (!ec && running_) do_private_connect();
    });
}

void CryptoClient::do_private_connect()
{
    std::cout << "[CryptoClient#" << instance_id_ << "] Starting private connect\n";

    auto timeout = std::make_shared<net::steady_timer>(*private_ioc_);
    timeout->expires_after(CONNECT_TIMEOUT);
    timeout->async_wait([this](auto ec) {
        if (!ec) handle_error("private connect timeout", net::error::timed_out);
    });

    private_resolver_->async_resolve(CR_PRIVATE_HOST, CR_SSL_PORT,
        [this, timeout](auto ec, tcp::resolver::results_type res) {
            if (ec) { timeout->cancel(); return handle_error("private resolve", ec); }

            auto stream = std::make_shared<beast::tcp_stream>(*private_ioc_);
            stream->expires_after(CONNECT_TIMEOUT);

            beast::get_lowest_layer(*stream).async_connect(res,
                [this, stream, timeout](auto ec, tcp::endpoint) {
                    if (ec) { timeout->cancel(); return handle_error("private connect", ec); }

                    auto ssl_stream = std::make_shared<ssl::stream<beast::tcp_stream>>(std::move(*stream), *ssl_ctx_);
                    if (!SSL_set_tlsext_host_name(ssl_stream->native_handle(), CR_PRIVATE_HOST)) {
                        return handle_error("SNI configuration",
                            beast::error_code(
                                static_cast<int>(::ERR_get_error()),
                                boost::asio::error::get_ssl_category()));
                    }
                    ssl_stream->async_handshake(ssl::stream_base::client,
                        [this, ssl_stream, timeout](auto ec) {
                            if (ec) { timeout->cancel(); return handle_error("private SSL", ec); }

                            {
			        SSL_set_mode(ssl_stream->native_handle(), SSL_MODE_AUTO_RETRY);
                                beast::get_lowest_layer(*ssl_stream).socket().set_option(boost::asio::socket_base::keep_alive(true));
                                std::lock_guard lq(private_mutex_);
                                private_ws_ = std::make_unique<websocket_stream>(std::move(*ssl_stream));
                                private_ws_->set_option(ws::stream_base::decorator(
                                    [](ws::request_type& req) {
                                        req.set(beast::http::field::user_agent, USER_AGENT);
                                        req.set(beast::http::field::host, CR_PRIVATE_HOST);
                                        req.set(beast::http::field::sec_websocket_protocol, "json");
                                    }));
                            }

                            auto response = std::make_shared<ws::response_type>();
                            private_ws_->async_handshake(*response, CR_PRIVATE_HOST, CR_PRIVATE_PATH,
                                [this, response, timeout](auto ec) {
                                    timeout->cancel();
                                    if (ec) {
                                        return handle_error("private handshake", ec, response->result_int());
                                    }
                                    std::cout << "[CryptoClient#" << instance_id_ << "] Private handshake completed\n";
                                    private_connected_ = true;
                                    authenticate();
                                    do_private_read();
                                });
                        });
                });
        });
}
/* ------------------------------------------------------------------ */
/*  Authentication & subscriptions                                    */
/* ------------------------------------------------------------------ */
void CryptoClient::authenticate()
{
    const long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    constexpr char METHOD[] = "public/auth";
    constexpr char ID[] = "1";

    json params = json::object();
    std::string payload = build_payload(METHOD, ID, api_key_, params, ts);

    unsigned char digest[32];
    unsigned int dlen{};
    HMAC(EVP_sha256(), api_secret_.data(), api_secret_.size(),
         reinterpret_cast<const unsigned char*>(payload.data()),
         payload.size(), digest, &dlen);

    std::ostringstream sig;
    sig << std::hex << std::setfill('0');
    for (unsigned i = 0; i < dlen; ++i)
        sig << std::setw(2) << static_cast<int>(digest[i]);

    json msg = {
        {"id", ID},
        {"method", METHOD},
        {"api_key", api_key_},
        {"sig", sig.str()},
        {"nonce", ts}
    };
    std::cout << "[CryptoClient#" << instance_id_ << "] Sending authentication: " << msg.dump() << '\n';
    send_private_msg(std::move(msg));
}

void CryptoClient::subscribe_public()
{
    json msg = {
        {"id", "11"},
        {"method", "subscribe"},
        {"params", {
			   {"channels", {"book." + symbol_ + ".10"}}, 
			   {"book_subscription_type", "SNAPSHOT"}
		   }}
    };

    std::cout << "[CryptoClient#" << instance_id_ << "] Sending public subscription: " << msg.dump() << '\n';
    send_public_msg(std::move(msg));
}

void CryptoClient::subscribe_private()
{
    send_private_msg({
        {"id", "12"},
        {"method", "subscribe"},
        {"params", {{"channels", {
            "user.trade." + symbol_,
            "user.order." + symbol_,
            "account"
        }}}}
    });
}

/* ------------------------------------------------------------------ */
/*  Messaging helpers                                                 */
/* ------------------------------------------------------------------ */
void CryptoClient::send_private_msg(json&& j)
{
    if (!private_q_.push(std::move(j))) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] private queue full\n";
        return;
    }
    if (private_ioc_) net::post(*private_ioc_, [this]{ write_next_private(); });
}

void CryptoClient::write_next_public()
{
    std::lock_guard lp(public_mutex_);
    if (!public_connected_ || public_writing_ || !public_ws_) return;

    json j;
    if (!public_q_.pop(j)) return;                   // nothing to send

    public_writing_ = true;
    const auto dump = j.dump();

    public_ws_->async_write(net::buffer(dump),
        [this](auto ec, std::size_t){
            {
                std::lock_guard lp(public_mutex_);
                public_writing_ = false;
            }
            if (ec) handle_error("public write", ec);
            else if (!public_q_.empty() && running_)
                net::post(*public_ioc_, [this]{ write_next_public(); });
        });
}

void CryptoClient::write_next_private()
{
    std::lock_guard lq(private_mutex_);
    if (!private_connected_ || private_writing_ || !private_ws_) return;

    json j;
    if (!private_q_.pop(j)) return;

    private_writing_ = true;
    const auto dump = j.dump();

    private_ws_->async_write(net::buffer(dump),
        [this](auto ec, std::size_t){
            {
                std::lock_guard lq(private_mutex_);
                private_writing_ = false;
            }
            if (ec) handle_error("private write", ec);
            else if (!private_q_.empty() && running_)
                net::post(*private_ioc_, [this]{ write_next_private(); });
        });
}

/* ------------------------------------------------------------------ */
/*  Read loops                                                        */
/* ------------------------------------------------------------------ */
void CryptoClient::do_public_read()
{
    std::lock_guard lp(public_mutex_);
    if (!public_ws_ || !public_connected_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Public read skipped: not connected\n";
        return;
    }

    auto timeout = std::make_shared<net::steady_timer>(*public_ioc_);
    timeout->expires_after(std::chrono::seconds(30));
    timeout->async_wait([this](auto ec) {
        if (!ec && running_) {
            std::lock_guard lp(public_mutex_);
            if (public_ws_) {
                // Send a WebSocket ping to keep the connection alive
                public_ws_->async_ping("keepalive",
                    [this](auto ec) {
                        if (ec) handle_error("public ping", ec);
                        else std::cout << "[CryptoClient#" << instance_id_ << "] Sent public ping\n";
                    });
                // Do not close the WebSocket; retry read
                if (running_) do_public_read();
            }
        }
    });

    public_ws_->async_read(public_buffer_,
        [this, timeout](auto ec, std::size_t bytes) {
            timeout->cancel();
            if (ec) {
                std::cerr << "[CryptoClient#" << instance_id_ << "] Public read error: " << ec.message() << ", bytes: " << bytes << '\n';
                return handle_error("public read", ec);
            }
            std::cout << "[CryptoClient#" << instance_id_ << "] Public read received: " << bytes << " bytes\n";
            json j = json::parse(beast::buffers_to_string(public_buffer_.data()), nullptr, false);
            if (j.is_discarded()) {
                std::cerr << "[CryptoClient#" << instance_id_ << "] Failed to parse public message: " << beast::buffers_to_string(public_buffer_.data()) << '\n';
            } else {
                std::cout << "[CryptoClient#" << instance_id_ << "] Public message: " << j.dump() << '\n';
                handle_public_msg(j);
            }
            public_buffer_.consume(public_buffer_.size());
            if (running_) do_public_read();
        });
}

void CryptoClient::do_private_read()
{
    std::lock_guard lq(private_mutex_);
    if (!private_ws_ || !private_connected_) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] Private read skipped: not connected\n";
        return;
    }

    auto timeout = std::make_shared<net::steady_timer>(*private_ioc_);
    timeout->expires_after(std::chrono::seconds(30));
    timeout->async_wait([this](auto ec) {
        if (!ec && running_) {
            std::lock_guard lq(private_mutex_);
            if (private_ws_) {
                private_ws_->async_ping("keepalive",
                    [this](auto ec) {
                        if (ec) handle_error("private ping", ec);
                        else std::cout << "[CryptoClient#" << instance_id_ << "] Sent private ping\n";
                    });
                if (running_) do_private_read();
            }
        }
    });

    private_ws_->async_read(private_buffer_,
        [this, timeout](auto ec, std::size_t bytes) {
            timeout->cancel();
            if (ec) {
                std::cerr << "[CryptoClient#" << instance_id_ << "] Private read error: " << ec.message() << ", bytes: " << bytes << '\n';
                return handle_error("private read", ec);
            }
            std::cout << "[CryptoClient#" << instance_id_ << "] Private read received: " << bytes << " bytes\n";
            json j = json::parse(beast::buffers_to_string(private_buffer_.data()), nullptr, false);
            if (j.is_discarded()) {
                std::cerr << "[CryptoClient#" << instance_id_ << "] Failed to parse private message: " << beast::buffers_to_string(private_buffer_.data()) << '\n';
            } else {
                std::cout << "[CryptoClient#" << instance_id_ << "] Private message: " << j.dump() << '\n';
                handle_private_msg(j);
            }
            private_buffer_.consume(private_buffer_.size());
            if (running_) do_private_read();
        });
}

/* ------------------------------------------------------------------ */
/*  Message dispatch                                                  */
/* ------------------------------------------------------------------ */
void CryptoClient::send_public_msg(json&& j)
{
    if (j.contains("method") && j["method"] == "public/respond-heartbeat") {
        std::lock_guard lp(public_mutex_);
        if (public_connected_ && public_ws_ && !public_writing_) {
            public_writing_ = true;
            const auto dump = j.dump();
            public_ws_->async_write(net::buffer(dump),
                [this](auto ec, std::size_t) {
                    std::lock_guard lp(public_mutex_);
                    public_writing_ = false;
                    if (ec) handle_error("public write", ec);
                    else if (!public_q_.empty() && running_)
                        net::post(*public_ioc_, [this]{ write_next_public(); });
                });
        }
        return;
    }

    if (!public_q_.push(std::move(j))) {
        std::cerr << "[CryptoClient#" << instance_id_ << "] public queue full\n";
        return;
    }
    if (public_ioc_) net::post(*public_ioc_, [this]{ write_next_public(); });
}

void CryptoClient::handle_public_msg(const json& j)
{
    std::cout << "[CryptoClient#" << instance_id_ << "] Handling public message: " << j.dump() << '\n';
    if (!j.contains("method")) return;

    const std::string method = j["method"];
    if (method == "public/heartbeat" || method == "heartbeat") {
        if (j.contains("id")) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Received public heartbeat, id: " << j["id"] << '\n';
            json response = {{"id", j["id"]}, {"method", "public/respond-heartbeat"}};
            std::cout << "[CryptoClient#" << instance_id_ << "] Sending public heartbeat response: " << response.dump() << '\n';
            send_public_msg(std::move(response));
        } else {
            std::cerr << "[CryptoClient#" << instance_id_ << "] Public heartbeat missing id: " << j.dump() << '\n';
        }
        return;
    }

    if (method == "subscribe" && orderbook_cb_ && j.contains("result") && j["result"].contains("data"))
        orderbook_cb_(j["result"]["data"]);
}

void CryptoClient::handle_private_msg(const json& j)
{
    std::cout << "[CryptoClient#" << instance_id_ << "] Handling private message: " << j.dump() << '\n';
    if (!j.contains("method")) return;

    const std::string method = j["method"];
    if (method == "public/heartbeat" || method == "heartbeat") {
        if (j.contains("id")) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Received private heartbeat, id: " << j["id"] << '\n';
            json response = {{"id", j["id"]}, {"method", "public/respond-heartbeat"}};
            std::cout << "[CryptoClient#" << instance_id_ << "] Sending private heartbeat response: " << response.dump() << '\n';
            send_private_msg(std::move(response));
        } else {
            std::cerr << "[CryptoClient#" << instance_id_ << "] Private heartbeat missing id: " << j.dump() << '\n';
        }
        return;
    }

    if (method == "public/auth") {
        if (j.value("code", -1) == 0) {
            std::cout << "[CryptoClient#" << instance_id_ << "] Authentication successful\n";
            subscribe_private();
        } else {
            std::cerr << "[CryptoClient#" << instance_id_ << "] auth failed: " << j.dump() << '\n';
        }
        return;
    }

    if (method.rfind("trade", 0) == 0 && trade_cb_)
        trade_cb_(j["result"]["data"]);
    else if (method.rfind("order", 0) == 0 && order_cb_)
        order_cb_(j["result"]["data"]);
    else if (method == "account" && position_cb_)
        position_cb_(j["result"]["data"]);
}

/* ------------------------------------------------------------------ */
/*  Accessor / callback setters                                       */
/* ------------------------------------------------------------------ */
void CryptoClient::set_orderbook_cb(std::function<void(const json&)> cb) { orderbook_cb_ = std::move(cb); }
void CryptoClient::set_trade_cb    (std::function<void(const json&)> cb) { trade_cb_     = std::move(cb); }
void CryptoClient::set_position_cb (std::function<void(const json&)> cb) { position_cb_  = std::move(cb); }
void CryptoClient::set_order_cb    (std::function<void(const json&)> cb) { order_cb_     = std::move(cb); }

/* ------------------------------------------------------------------ */
/*  Trading helpers                                                   */
/* ------------------------------------------------------------------ */
void CryptoClient::place_order(const std::string& side,double price,double size,
                               const std::string& client_oid,bool hedge,
                               const std::string& type)
{
    send_private_msg({
        {"id", "20"},
        {"method", "private/create-order"},
        {"params", {
            {"instrument_name", hedge ? hedge_symbol_ : symbol_},
            {"side", side},
            {"type", type},
            {"price", price},
            {"quantity", size},
            {"client_oid", client_oid}
        }}
    });
}

void CryptoClient::cancel_order(const std::string& order_id)
{
    send_private_msg({
        {"id", "21"},
        {"method", "private/cancel-order"},
        {"params", {{"order_id", order_id}}}
    });
}

void CryptoClient::cancel_all_orders()
{
    send_private_msg({
        {"id", "22"},
        {"method", "private/cancel-all-orders"},
        {"params", {{"instrument_name", symbol_}}}
    });
}

void CryptoClient::get_position()
{
    send_private_msg({
        {"id", "23"},
        {"method", "private/get-positions"},
        {"params", {{"instrument_name", symbol_}}}
    });
}

/* ------------------------------------------------------------------ */
/*  Error handler                                                     */
/* ------------------------------------------------------------------ */
void CryptoClient::handle_error(const std::string& where,
                                const beast::error_code& ec,
                                int http_status)
{
    std::cerr << "[CryptoClient#" << instance_id_ << "] " << where
              << " : " << ec.message() << '\n';

    /* retry on connection-level issues ----------------------------- */
    if (ec == net::error::connection_refused ||
        ec == net::error::timed_out           ||
        ec == ssl::error::stream_truncated)
    {
        auto delay = std::chrono::milliseconds(
            http_status == 429 ? 60'000 : 3'000 * (1 + retry_count_++));

        std::cerr << "[CryptoClient#" << instance_id_ << "] retry " << where
                  << " in " << delay.count() << "ms\n";

        if (where.find("public") != std::string::npos) {
            if (public_timer_) {
                public_timer_->expires_after(delay);
                public_timer_->async_wait([this](auto ec){
                    if (!ec && running_) do_public_connect();
                });
            }
        } else {
            if (private_timer_) {
                private_timer_->expires_after(delay);
                private_timer_->async_wait([this](auto ec){
                    if (!ec && running_) do_private_connect();
                });
            }
        }
        return;
    }

    retry_count_ = 0;    // non-retryable error: reset back-off
}

} // namespace RLTrader
