#include "crypto_client.h"
#include <boost/asio/connect.hpp>
#include <boost/asio/post.hpp>
#include <iomanip>
#include <iostream>

namespace RLTrader {

//constexpr const char *CR_PUBLIC_HOST  = "stream.crypto.com";
//constexpr const char *CR_PRIVATE_HOST = "stream.crypto.com";
constexpr const char *CR_PUBLIC_HOST  = "uat-stream.3ona.co";
constexpr const char *CR_PRIVATE_HOST = "uat-stream.3ona.co";
constexpr const char *CR_PUBLIC_PATH  = "/exchange/v1/market";
constexpr const char *CR_PRIVATE_PATH = "/exchange/v1/user";
constexpr const char *CR_SSL_PORT     = "443";

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
    if (running_.exchange(true)) return;

    public_ioc_  = std::make_unique<net::io_context>();
    private_ioc_ = std::make_unique<net::io_context>();
    ssl_ctx_     = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
    ssl_ctx_->set_default_verify_paths();
    ssl_ctx_->set_verify_mode(ssl::verify_peer);

    public_resolver_  = std::make_unique<tcp::resolver>(*public_ioc_);
    private_resolver_ = std::make_unique<tcp::resolver>(*private_ioc_);

    public_work_  = std::make_unique<boost::asio::executor_work_guard<net::io_context::executor_type>>(public_ioc_->get_executor());
    private_work_ = std::make_unique<boost::asio::executor_work_guard<net::io_context::executor_type>>(private_ioc_->get_executor());

    setup_connections();

    public_thread_  = std::make_unique<std::thread>([this] { public_ioc_->run(); });
    private_thread_ = std::make_unique<std::thread>([this] { private_ioc_->run(); });
}

void CryptoClient::stop() {
    if (!running_.exchange(false)) return;

    boost::system::error_code ec;
    if (public_ws_  && public_connected_)  public_ws_->close(websocket::close_code::normal, ec);
    if (private_ws_ && private_connected_) private_ws_->close(websocket::close_code::normal, ec);

    if (public_work_)  public_work_.reset();
    if (private_work_) private_work_.reset();

    if (public_ioc_)  public_ioc_->stop();
    if (private_ioc_) private_ioc_->stop();

    if (public_thread_  && public_thread_->joinable())  public_thread_->join();
    if (private_thread_ && private_thread_->joinable()) private_thread_->join();
}

/* ---------------------- connection helpers --------------------------- */
void CryptoClient::setup_connections() {
    setup_public_ws();
    setup_private_ws();
}

void CryptoClient::setup_public_ws() {
    public_ws_   = std::make_unique<websocket_stream>(*public_ioc_, *ssl_ctx_);
    public_timer_= std::make_unique<net::steady_timer>(*public_ioc_);
    public_timer_->expires_after(std::chrono::milliseconds(500));
    public_timer_->async_wait([this](auto ec){ if(!ec && running_) do_public_connect(); });
}

void CryptoClient::setup_private_ws() {
    private_ws_   = std::make_unique<websocket_stream>(*private_ioc_, *ssl_ctx_);
    private_timer_= std::make_unique<net::steady_timer>(*private_ioc_);
    private_timer_->expires_after(std::chrono::milliseconds(500));
    private_timer_->async_wait([this](auto ec){ if(!ec && running_) do_private_connect(); });
}

void CryptoClient::do_public_connect() {
    public_resolver_->async_resolve(CR_PUBLIC_HOST, CR_SSL_PORT,
        [this](auto ec, tcp::resolver::results_type res){
            if (ec) return handle_error("Public resolve", ec);

            boost::asio::async_connect(beast::get_lowest_layer(*public_ws_), res,
                [this](auto ec, tcp::endpoint){
                    if (ec) return handle_error("Public connect", ec);

                    public_ws_->next_layer().async_handshake(ssl::stream_base::client,
                        [this](auto ec){
                            if (ec) return handle_error("Public SSL", ec);

                            public_ws_->async_handshake(CR_PUBLIC_HOST, CR_PUBLIC_PATH,
                                [this](auto ec){
                                    if (ec) return handle_error("Public WS", ec);
                                    public_connected_ = true;
                                    subscribe_public();
                                    do_public_read();
                                });
                        });
                });
        });
}

void CryptoClient::do_private_connect() {
    private_resolver_->async_resolve(CR_PRIVATE_HOST, CR_SSL_PORT,
        [this](auto ec, tcp::resolver::results_type res){
            if (ec) return handle_error("Private resolve", ec);

            boost::asio::async_connect(beast::get_lowest_layer(*private_ws_), res,
                [this](auto ec, tcp::endpoint){
                    if (ec) return handle_error("Private connect", ec);

                    private_ws_->next_layer().async_handshake(ssl::stream_base::client,
                        [this](auto ec){
                            if (ec) return handle_error("Private SSL", ec);

                            private_ws_->async_handshake(CR_PRIVATE_HOST, CR_PRIVATE_PATH,
                                [this](auto ec){
                                    if (ec) return handle_error("Private WS", ec);
                                    private_connected_ = true;
                                    authenticate();
                                    do_private_read();
                                });
                        });
                });
        });
}

/* ---------------------- authentication & subs ----------------------- */
void CryptoClient::authenticate() {
    long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
    std::string method = "public/auth";
    std::string id = "1";

    std::ostringstream prehash;
    prehash << method << id << api_key_ << ts;
    unsigned char digest[32];
    HMAC(EVP_sha256(),
         api_secret_.data(), api_secret_.size(),
         reinterpret_cast<const unsigned char*>(prehash.str().data()), prehash.str().size(),
         digest, nullptr);

    std::ostringstream sig;
    sig << std::hex << std::setfill('0');
    for (int i=0;i<32;++i) sig << std::setw(2) << static_cast<int>(digest[i]);

    send_private_msg({
        {"id", id},
        {"method", method},
        {"api_key", api_key_},
        {"sig", sig.str()},
        {"nonce", ts}
    });
}

void CryptoClient::subscribe_public() {
    send_public_msg({
        {"id", "11"},
        {"method", "subscribe"},
        {"params", {{"channels", {"orderbook." + symbol_ + ".20"}}}}
    });
}

void CryptoClient::subscribe_private() {
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
    if (!public_q_.push(std::move(j))) return;
    net::post(*public_ioc_, [this]{ write_next_public(); });
}
void CryptoClient::send_private_msg(json&& j) {
    if (!private_q_.push(std::move(j))) return;
    net::post(*private_ioc_, [this]{ write_next_private(); });
}

void CryptoClient::write_next_public() {
    if (!public_connected_ || public_writing_) return;
    json j;
    if (public_q_.pop(j)) {
        public_writing_ = true;
        auto dump = j.dump();
        public_ws_->async_write(net::buffer(dump),
            [this](auto ec, std::size_t){
                public_writing_ = false;
                if (ec) handle_error("Public write", ec);
                else if (!public_q_.empty()) write_next_public();
            });
    }
}
void CryptoClient::write_next_private() {
    if (!private_connected_ || private_writing_) return;
    json j;
    if (private_q_.pop(j)) {
        private_writing_ = true;
        auto dump = j.dump();
        private_ws_->async_write(net::buffer(dump),
            [this](auto ec, std::size_t){
                private_writing_ = false;
                if (ec) handle_error("Private write", ec);
                else if (!private_q_.empty()) write_next_private();
            });
    }
}

/* --------------------------- read loops ----------------------------- */
void CryptoClient::do_public_read() {
    public_ws_->async_read(public_buffer_,
        [this](auto ec, std::size_t){
            if (ec) return handle_error("Public read", ec);
            json j = json::parse(beast::buffers_to_string(public_buffer_.data()), nullptr, false);
            public_buffer_.consume(public_buffer_.size());
            if (!j.is_discarded()) handle_public_msg(j);
            if (running_) do_public_read();
        });
}
void CryptoClient::do_private_read() {
    private_ws_->async_read(private_buffer_,
        [this](auto ec, std::size_t){
            if (ec) return handle_error("Private read", ec);
            json j = json::parse(beast::buffers_to_string(private_buffer_.data()), nullptr, false);
            private_buffer_.consume(private_buffer_.size());
            if (!j.is_discarded()) handle_private_msg(j);
            if (running_) do_private_read();
        });
}

/* ---------------------- message dispatchers ------------------------- */
void CryptoClient::handle_public_msg(const json& j) {
    if (!j.contains("method") || j["method"] != "subscribe") return;
    const auto& data = j["result"]["data"];
    if (orderbook_cb_) orderbook_cb_(data);
}

void CryptoClient::handle_private_msg(const json& j) {
    if (j.contains("method") && j["method"] == "public/auth") {
        if (j.contains("code") && j["code"] == 0) subscribe_private();
        return;
    }
    if (!j.contains("method")) return;
    const std::string m = j["method"];
    const auto& data   = j["result"]["data"];
    if (m.rfind("trade",0)==0 && trade_cb_) trade_cb_(data);
    else if (m.rfind("order",0)==0 && order_cb_) order_cb_(data);
    else if (m == "account"   && position_cb_) position_cb_(data);
}

/* ---------------- accessors / callbacks ----------------------------- */
void CryptoClient::set_orderbook_cb(std::function<void(const json&)> cb){ orderbook_cb_ = std::move(cb); }
void CryptoClient::set_trade_cb    (std::function<void(const json&)> cb){ trade_cb_     = std::move(cb); }
void CryptoClient::set_position_cb (std::function<void(const json&)> cb){ position_cb_  = std::move(cb); }
void CryptoClient::set_order_cb    (std::function<void(const json&)> cb){ order_cb_     = std::move(cb); }

/* ---------------- rest of trading helpers --------------------------- */
void CryptoClient::place_order(const std::string& side,double price,double size,
                               const std::string& client_oid,bool is_hedge,const std::string& type) {
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
    send_private_msg({ {"id","21"},{"method","private/cancel-order"},{"params",{{"order_id",order_id}}} });
}
void CryptoClient::cancel_all_orders() {
    send_private_msg({ {"id","22"},{"method","private/cancel-all-orders"},{"params",{{"instrument_name",symbol_}}} });
}
void CryptoClient::get_position() {
    send_private_msg({ {"id","23"},{"method","private/get-positions"},{"params",{{"instrument_name",symbol_}}} });
}

/* --------------------------- error handler -------------------------- */
void CryptoClient::handle_error(const std::string& where, const beast::error_code& ec) {
    std::cerr << "[CryptoClient#" << instance_id_ << "] " << where << " : " << ec.message() << '\n';
    if (ec.message() == "Too many requests") {
        // Implement exponential backoff for rate limit errors
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if (where.find("Public") != std::string::npos) {
            public_timer_->expires_after(std::chrono::milliseconds(2000));
            public_timer_->async_wait([this](auto ec){ if(!ec && running_) do_public_connect(); });
        } else {
            private_timer_->expires_after(std::chrono::milliseconds(2000));
            private_timer_->async_wait([this](auto ec){ if(!ec && running_) do_private_connect(); });
        }
    }
}

} // namespace RLTrader
