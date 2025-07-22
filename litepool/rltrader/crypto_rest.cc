#include "crypto_rest.h"
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

constexpr const char* REST_HOST   = "https://api.crypto.com/exchange/v1/";
constexpr const char* REST_PORT   = "443";
constexpr const char* USER_AGENT  = "AlphaQraft_Trading";
static constexpr std::chrono::milliseconds REQUEST_DELAY{10'000};
static constexpr std::chrono::seconds REQUEST_TIMEOUT{5};

/* ------------------------------------------------------------------ */
// Utility function to match Python's params_to_str
std::string build_payload(const std::string& method, int id, const std::string& api_key,
                         const json& params, long nonce) {
    std::string param_string;
    if (!params.empty()) {
        std::map<std::string, std::string> sorted_params;
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.value().is_string()) {
                sorted_params[it.key()] = it.value().get<std::string>();
            } else if (it.value().is_null()) {
                sorted_params[it.key()] = "null";
            } else if (it.value().is_array()) {
                std::string array_str;
                for (const auto& item : it.value()) {
                    if (item.is_string()) {
                        array_str += item.get<std::string>();
                    } else {
                        array_str += item.dump();
                    }
                }
                sorted_params[it.key()] = array_str;
            } else {
                sorted_params[it.key()] = it.value().dump();
            }
        }
        for (const auto& [key, value] : sorted_params) {
            param_string += key + value;
        }
    }
    std::string payload = method + std::to_string(id) + api_key + param_string + std::to_string(nonce);
    std::cout << "[CryptoREST] Signing string: " << payload << "\n";
    return payload;
}

/* ------------------------------------------------------------------ */
CryptoREST::CryptoREST(const std::string& api_key,
                       const std::string& api_secret)
    : api_key_(api_key), api_secret_(api_secret)
{
    try {
        ssl_ctx_.set_default_verify_paths();
        ssl_ctx_.set_verify_mode(ssl::verify_peer);
        SSL_CTX_set_options(ssl_ctx_.native_handle(), SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);
        SSL_CTX_set_cipher_list(ssl_ctx_.native_handle(), "HIGH:!aNULL:!kRSA:!PSK:!SRP:!MD5:!RC4");
        std::cout << "[CryptoREST] SSL context initialized\n";
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
        auto results = resolver.resolve(REST_HOST, REST_PORT);

        net::connect(socket_->next_layer(), results.begin(), results.end());
        
        if (!SSL_set_tlsext_host_name(socket_->native_handle(), REST_HOST)) {
	    std::cout << "REST connect failed " << std::endl;
            boost::system::error_code ec{static_cast<int>(::ERR_get_error()),
                                         net::error::get_ssl_category()};
            throw boost::system::system_error{ec};
        }

        socket_->handshake(ssl::stream_base::client);
        std::cout << "[CryptoREST] Connected to " << REST_HOST << ":" << REST_PORT << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[CryptoREST] connect: " << e.what() << '\n';
        socket_.reset();
    }
}

/* ------------------------------------------------------------------ */
bool CryptoREST::fetch_position(const std::string& symbol,
                                double& amount,
                                double& avg_price)
{
    amount = 0.0;
    avg_price = 0.0;
    std::cout << "REST fetching positions" << std::endl;

    std::lock_guard lk(connection_mutex_);
    if (!socket_) do_connect();
    if (!socket_) { std::cerr << "Failed to connect REST" << std::endl; return false; }

    try {
        /* ---- build authenticated body -------------------------------- */
        const long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now()
                                .time_since_epoch())
                            .count();

        constexpr char METHOD[] = "private/get-positions";
        constexpr int ID = 1;

        json params = {{"instrument_name", symbol}};
        std::string payload = build_payload(METHOD, ID, api_key_, params, ts);

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
        std::cout << "[CryptoREST] Signature: " << sig.str() << "\n";

        json body = {
            {"id", ID},
            {"method", METHOD},
            {"api_key", api_key_},
            {"params", params},
            {"nonce", ts},
            {"sig", sig.str()}
        };

        /* ---- send ---------------------------------------------------- */
        http::request<http::string_body> req{http::verb::post,
                                             "private/get-positions",
                                             11};
        req.set(http::field::host, REST_HOST);
        req.set(http::field::user_agent, USER_AGENT);
        req.set(http::field::content_type, "application/json");
        req.body() = body.dump();
        req.prepare_payload();

        std::cout << "[CryptoREST] Sending request: " << body.dump() << "\n";
        http::write(*socket_, req);

        /* ---- receive ------------------------------------------------- */
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        
        auto timeout = std::make_shared<net::steady_timer>(ioc_);
        timeout->expires_after(REQUEST_TIMEOUT);
        timeout->async_wait([&](const boost::system::error_code& ec) {
            if (!ec) {
                boost::system::error_code cancel_ec;
                socket_->next_layer().cancel(cancel_ec);
                if (cancel_ec) {
                    std::cerr << "[CryptoREST] Timeout cancel error: " << cancel_ec.message() << '\n';
                }
            }
        });

        http::read(*socket_, buffer, res);
        timeout->cancel();

        /* ---- parse --------------------------------------------------- */
        if (res.result() != http::status::ok) {
            std::cerr << "[CryptoREST] HTTP " << res.result_int()
                      << " : " << res.reason() << " - " << res.body() << '\n';
            socket_.reset();
            return false;
        }

        json j = json::parse(res.body(), nullptr, false);
        if (j.is_discarded() || !j.contains("result") || !j["result"].contains("data")) {
            std::cerr << "[CryptoREST] Bad JSON: " << res.body() << '\n';
            socket_.reset();
            return false;
        }

        std::cout << "[CryptoREST] Response: " << j.dump() << "\n";

        for (const auto& p : j["result"]["data"]) {
            if (p.contains("instrument_name") && p["instrument_name"] == symbol) {
                amount = p.value("quantity", 0.0);
                avg_price = std::abs(p.value("open_pos_cost", 0.0));
                break;
            }
        }

        /* ---- cooldown + cleanup ------------------------------------- */
        boost::system::error_code ec;
        socket_->next_layer().shutdown(tcp::socket::shutdown_both, ec);
        if (ec) std::cerr << "[CryptoREST] Shutdown error: " << ec.message() << '\n';
        socket_->next_layer().close(ec);
        if (ec) std::cerr << "[CryptoREST] Close error: " << ec.message() << '\n';
        socket_.reset();
        std::this_thread::sleep_for(REQUEST_DELAY);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[CryptoREST] fetch_position: " << e.what() << '\n';
        socket_.reset();
        return false;
    }
}

} // namespace RLTrader
