#include "crypto_rest.h"
#include "crypto_util.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include <boost/asio/connect.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <openssl/evp.h>
#include <openssl/hmac.h>

namespace RLTrader
{
    namespace beast = boost::beast;
    namespace http  = beast::http;
    namespace net   = boost::asio;
    namespace ssl   = net::ssl;
    using tcp       = net::ip::tcp;

/* ------------------------------------------------------------------ */

//constexpr const char* REST_HOST   = "api.crypto.com";
constexpr const char* REST_HOST   = "uat-api.3ona.co";
constexpr const char* REST_PORT   = "443";
constexpr const char* USER_AGENT  = "AlphaQraft_Trading";
static   constexpr std::chrono::milliseconds REQUEST_DELAY{10'000};
static   constexpr std::chrono::seconds      REQUEST_TIMEOUT{5};

/* ------------------------------------------------------------------ */
CryptoREST::CryptoREST(const std::string& api_key,
                       const std::string& api_secret)
    : api_key_(api_key), api_secret_(api_secret)
{
    try {
        ssl_ctx_.set_default_verify_paths();
        ssl_ctx_.set_verify_mode(ssl::verify_peer);
        std::cout << "[CryptoREST] SSL context initialised\n";
    } catch (const std::exception& e) {
        std::cerr << "[CryptoREST] SSL context error: " << e.what() << '\n';
    }
}

CryptoREST::~CryptoREST()
{
    std::lock_guard lk(connection_mutex_);
    if (socket_) {
        boost::system::error_code ec;
        socket_->next_layer().close(ec);
        if (ec) std::cerr << "[CryptoREST] socket close: " << ec.message() << '\n';
    }
    std::cout << "[CryptoREST] destroyed\n";
}

/* ------------------------------------------------------------------ */
void CryptoREST::do_connect()
{
    std::lock_guard lk(connection_mutex_);

    try {
        if (!socket_)
            socket_ = std::make_unique<ssl::stream<tcp::socket>>(ioc_, ssl_ctx_);

        tcp::resolver resolver(ioc_);
        auto          results = resolver.resolve(REST_HOST, REST_PORT);

        net::connect(socket_->next_layer(), results.begin(), results.end());
        socket_->handshake(ssl::stream_base::client);
    } catch (const std::exception& e) {
        std::cerr << "[CryptoREST] connect: " << e.what() << '\n';
        socket_.reset();
    }
}

/* ------------------------------------------------------------------ */
bool CryptoREST::fetch_position(const std::string& symbol,
                                double&            amount,
                                double&            avg_price)
{
    amount     = 0.0;
    avg_price  = 0.0;

    std::lock_guard lk(connection_mutex_);
    if (!socket_) do_connect();
    if (!socket_) return false;

    try {
        /* ---- build authenticated body -------------------------------- */
        const long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now()
                                .time_since_epoch())
                            .count();

        constexpr char METHOD[] = "private/get-positions";
        constexpr char ID[]     = "1";

        json params = {{"instrument_name", symbol}};
        std::string payload = build_payload(METHOD, ID, api_key_, params, ts);

        unsigned char digest[32]; unsigned int dlen{};
        HMAC(EVP_sha256(), api_secret_.data(), api_secret_.size(),
             reinterpret_cast<const unsigned char*>(payload.data()),
             payload.size(), digest, &dlen);

        std::ostringstream sig;
        sig << std::hex << std::setfill('0');
        for (unsigned i = 0; i < dlen; ++i)
            sig << std::setw(2) << static_cast<int>(digest[i]);

        json body = {
            {"id",      ID},
            {"method",  METHOD},
            {"api_key", api_key_},
            {"params",  params},
            {"nonce",   ts},
            {"sig",     sig.str()}
        };

        /* ---- send ---------------------------------------------------- */
        http::request<http::string_body> req{http::verb::post,
                                             "/v2/private/get-positions",
                                             11};
        req.set(http::field::host,        REST_HOST);
        req.set(http::field::user_agent,  USER_AGENT);
        req.set(http::field::content_type,"application/json");
        req.body() = body.dump();
        req.prepare_payload();

        http::write(*socket_, req);

        /* ---- receive ------------------------------------------------- */
        beast::flat_buffer             buf;
        http::response<http::string_body> res;
        http::read(*socket_, buf, res);

        /* ---- parse --------------------------------------------------- */
        if (res.result() != http::status::ok) {
            std::cerr << "[CryptoREST] HTTP " << res.result_int()
                      << " : " << res.reason() << '\n';
            return false;
        }

        json j = json::parse(res.body(), nullptr, /*allow_exceptions*/false);
        if (j.is_discarded() ||
            !j.contains("result") || !j["result"].contains("data"))
        {
            std::cerr << "[CryptoREST] bad JSON: " << res.body() << '\n';
            return false;
        }

        for (const auto& p : j["result"]["data"]) {
            if (p.contains("instrument_name") &&
                p["instrument_name"] == symbol)
            {
                amount     = p.value("quantity",  0.0);
                avg_price  = p.value("avg_price", 0.0);
                break;
            }
        }

        /* ---- cooldown + cleanup ------------------------------------- */
        boost::system::error_code ec;
        socket_->next_layer().shutdown(tcp::socket::shutdown_both, ec);
        socket_->next_layer().close(ec);
        socket_.reset();
        std::this_thread::sleep_for(REQUEST_DELAY);

        return true;
    } catch (const std::exception& e) {
        std::cout << "[CryptoREST] fetch_position: " << e.what() << '\n';
        socket_.reset();
        return false;
    }
}

} // namespace RLTrader
