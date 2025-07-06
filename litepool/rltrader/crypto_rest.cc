#include "crypto_rest.h"
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>

namespace RLTrader {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;

//constexpr const char* REST_HOST = "api.crypto.com";
constexpr const char* REST_HOST = "uat-api.3ona.co";
constexpr const char* REST_PORT = "443";
constexpr const char* USER_AGENT = "RLTrader/1.0 (Crypto.com V1 API Client)";
static constexpr std::chrono::milliseconds REQUEST_DELAY{10000}; // 10s delay to avoid rate limits

/* --------------------------------------------------------------------- */
CryptoREST::CryptoREST(const std::string& api_key, const std::string& api_secret)
    : api_key_(api_key), api_secret_(api_secret), ioc_(), ssl_ctx_(ssl::context::tlsv12_client) {
    try {
        ssl_ctx_.set_default_verify_paths();
        ssl_ctx_.set_verify_mode(ssl::verify_peer);
        std::cout << "[CryptoREST] Initialized SSL context\n";
    } catch (const std::exception& e) {
        std::cerr << "[CryptoREST] SSL context initialization failed: " << e.what() << '\n';
    }
}

CryptoREST::~CryptoREST() {
    boost::system::error_code ec;
    if (socket_) {
        socket_->next_layer().close(ec);
        if (ec) std::cerr << "[CryptoREST] Socket close error: " << ec.message() << '\n';
        socket_.reset();
    }
    std::cout << "[CryptoREST] Destroyed\n";
}

/* --------------------------------------------------------------------- */
bool CryptoREST::fetch_position(const std::string& symbol, double& amount, double& avg_price) {
    amount = avg_price = 0; // Initialize to safe defaults
    try {
        // Initialize socket if not already done
        if (!socket_) {
            socket_ = std::make_unique<ssl::stream<tcp::socket>>(ioc_, ssl_ctx_);
            std::cout << "[CryptoREST] Created new SSL socket\n";
        }

        // Resolve host
        tcp::resolver resolver(ioc_);
        auto const results = resolver.resolve(REST_HOST, REST_PORT);
        std::cout << "[CryptoREST] Resolved " << REST_HOST << ":" << REST_PORT << "\n";

        // Connect
        boost::asio::connect(socket_->next_layer(), results.begin(), results.end());
        std::cout << "[CryptoREST] Connected to " << REST_HOST << "\n";

        // Perform SSL handshake
        socket_->handshake(ssl::stream_base::client);
        std::cout << "[CryptoREST] SSL handshake completed\n";

        // Construct request
        long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count();
        std::string method = "private/get-positions";
        std::string id = "1";

        std::ostringstream prehash;
        prehash << method << id << api_key_ << "instrument_name" << symbol << ts;
        unsigned char digest[32];
        unsigned int digest_len;
        HMAC(EVP_sha256(),
             api_secret_.data(), api_secret_.size(),
             reinterpret_cast<const unsigned char*>(prehash.str().data()), prehash.str().size(),
             digest, &digest_len);

        std::ostringstream sig;
        sig << std::hex << std::setfill('0');
        for (unsigned int i = 0; i < digest_len; ++i) sig << std::setw(2) << static_cast<int>(digest[i]);

        json req_body = {
            {"id", id},
            {"method", method},
            {"params", {{"instrument_name", symbol}}},
            {"api_key", api_key_},
            {"sig", sig.str()},
            {"nonce", ts}
        };

        http::request<http::string_body> req{http::verb::post, "/v2/private/get-positions", 11};
        req.set(http::field::host, REST_HOST);
        req.set(http::field::user_agent, USER_AGENT);
        req.set(http::field::content_type, "application/json");
        req.body() = req_body.dump();
        req.prepare_payload();

        // Send request
        http::write(*socket_, req);
        std::cout << "[CryptoREST] Sent request: " << req_body.dump() << "\n";

        // Receive response
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(*socket_, buffer, res);
        std::cout << "[CryptoREST] Received response: HTTP " << res.result_int() << " " << res.reason() << "\n";

        // Close socket to avoid reuse issues
        boost::system::error_code ec;
        socket_->next_layer().close(ec);
        if (ec) std::cerr << "[CryptoREST] Socket close error: " << ec.message() << '\n';
        socket_.reset();

        // Apply delay to avoid rate limits
        std::this_thread::sleep_for(REQUEST_DELAY);

        // Parse response
        json j = json::parse(res.body(), nullptr, false);
        if (j.is_discarded()) {
            std::cerr << "[CryptoREST] Failed to parse response: " << res.body() << "\n";
            return false;
        }

        if (res.result() != http::status::ok) {
            std::cerr << "[CryptoREST] HTTP error: " << res.result_int() << " " << res.reason() << "\n";
            return false;
        }

        if (j.contains("result") && j["result"].contains("data")) {
            const auto& data = j["result"]["data"];
            if (!data.is_array()) {
                std::cerr << "[CryptoREST] Invalid data format: " << j.dump() << "\n";
                return false;
            }
            for (const auto& pos : data) {
                if (pos.contains("instrument_name") && pos["instrument_name"] == symbol) {
                    amount = pos.contains("quantity") ? pos["quantity"].get<double>() : 0.0;
                    avg_price = pos.contains("avg_price") ? pos["avg_price"].get<double>() : 0.0;
                    std::cout << "[CryptoREST] Position fetched: symbol=" << symbol
                              << ", amount=" << amount << ", avg_price=" << avg_price << "\n";
                    return true;
                }
            }
            std::cout << "[CryptoREST] No position found for symbol: " << symbol << "\n";
            return false;
        } else {
            std::cerr << "[CryptoREST] Invalid response structure: " << j.dump() << "\n";
            return false;
        }
    } catch (const boost::system::system_error& e) {
        std::cerr << "[CryptoREST] Error: " << e.what() << " [" << e.code().message() << "]\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(15000)); // Longer backoff
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[CryptoREST] Exception: " << e.what() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(15000));
        return false;
    }
}

} // namespace RLTrader
