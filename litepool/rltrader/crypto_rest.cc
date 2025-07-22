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
    namespace http = beast::http;
    namespace net = boost::asio;
    namespace ssl = net::ssl;
    using tcp = net::ip::tcp;

    constexpr const char* REST_HOST = "api.crypto.com";
    constexpr const char* REST_PORT = "443";
    constexpr const char* API_BASE_PATH = "/exchange/v1/";
    constexpr const char* USER_AGENT = "AlphaQraft_Trading";
    static constexpr std::chrono::milliseconds REQUEST_DELAY{10000};
    static constexpr std::chrono::seconds REQUEST_TIMEOUT{5};

    std::string build_payload(const std::string& method, int id, const std::string& api_key,
                             const json& params, long nonce) 
    {
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

    CryptoREST::CryptoREST(const std::string& api_key,
                           const std::string& api_secret)
        : api_key_(api_key), api_secret_(api_secret)
    {
        try {
            ssl_ctx_.set_default_verify_paths();
            ssl_ctx_.set_verify_mode(ssl::verify_peer);
            ssl_ctx_.set_verify_callback(ssl::host_name_verification(REST_HOST));
            SSL_CTX_set_options(ssl_ctx_.native_handle(), 
                SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);
            SSL_CTX_set_cipher_list(ssl_ctx_.native_handle(), 
                "HIGH:!aNULL:!kRSA:!PSK:!SRP:!MD5:!RC4");
        } catch (const std::exception& e) {
            std::cerr << "[CryptoREST] SSL context error: " << e.what() << '\n';
            throw;
        }
    }

    CryptoREST::~CryptoREST()
    {
        running_ = false;
        std::lock_guard lk(connection_mutex_);
        if (socket_) {
            boost::system::error_code ec;
            socket_->next_layer().close(ec);
        }
    }

void CryptoREST::do_connect()
{
    const int max_retries = 3;
    boost::system::error_code ec;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        try {
            std::cout << "[CryptoREST] Connection attempt " << (attempt+1) << "/" << max_retries << "\n";

            // Create fresh socket
            socket_ = std::make_unique<ssl::stream<tcp::socket>>(ioc_, ssl_ctx_);
            socket_->next_layer().open(tcp::v4());

            // Set TCP_NODELAY option
            socket_->next_layer().set_option(tcp::no_delay(true));

            // For timeouts, we'll use async operations with timers instead
            // of socket-level timeouts which aren't available in Boost.Asio

            // Resolve endpoint
            tcp::resolver resolver(ioc_);
            auto endpoints = resolver.resolve(REST_HOST, REST_PORT, ec);
            if (ec) throw boost::system::system_error(ec);

            // Connect with deadline timer
            net::steady_timer timer(ioc_);
            timer.expires_after(REQUEST_TIMEOUT);

            bool timed_out = false;
            timer.async_wait([&](const boost::system::error_code&) {
                timed_out = true;
                socket_->next_layer().close();
            });

            net::async_connect(
                socket_->next_layer(),
                endpoints,
                [&](const boost::system::error_code& err, const tcp::endpoint&) {
                    timer.cancel();
                    ec = err;
                });

            ioc_.run();
            ioc_.restart();

            if (timed_out) {
                throw boost::system::system_error(net::error::timed_out);
            }
            if (ec) {
                throw boost::system::system_error(ec);
            }

            // SSL handshake
            if (!SSL_set_tlsext_host_name(socket_->native_handle(), REST_HOST)) {
                throw boost::system::system_error(
                    boost::system::error_code(
                        static_cast<int>(::ERR_get_error()),
                        net::error::get_ssl_category()));
            }

            // Set handshake timeout
            timer.expires_after(REQUEST_TIMEOUT);
            timed_out = false;
            timer.async_wait([&](const boost::system::error_code&) {
                timed_out = true;
                socket_->next_layer().close();
            });

            socket_->async_handshake(ssl::stream_base::client,
                [&](const boost::system::error_code& err) {
                    timer.cancel();
                    ec = err;
                });

            ioc_.run();
            ioc_.restart();

            if (timed_out) {
                throw boost::system::system_error(net::error::timed_out);
            }
            if (ec) {
                throw boost::system::system_error(ec);
            }

            std::cout << "[CryptoREST] Connected successfully\n";
            return;

        } catch (const std::exception& e) {
            std::cerr << "[CryptoREST] Attempt failed: " << e.what() << "\n";
            if (socket_) socket_->next_layer().close();
            if (attempt == max_retries-1) throw;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}
    bool CryptoREST::ensure_connection()
    {
        try {
            std::lock_guard lk(connection_mutex_);
            if (!socket_) {
                do_connect();
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[CryptoREST] Connection failed: " << e.what() << "\n";
            return false;
        }
    }

    bool CryptoREST::fetch_position(const std::string& symbol,
                                    double& amount,
                                    double& avg_price)
    {
        amount = 0.0;
        avg_price = 0.0;

        if (!ensure_connection()) {
            return false;
        }

        try {
            // Prepare request
            const long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            const std::string method = "private/get-positions";
            const std::string full_path = API_BASE_PATH + method;
            const int id = 1;

            json params = {{"instrument_name", symbol}};
            std::string payload = build_payload(method, id, api_key_, params, ts);

            // Generate signature
            unsigned char digest[32];
            unsigned int dlen;
            HMAC(EVP_sha256(), api_secret_.data(), api_secret_.size(),
                 reinterpret_cast<const unsigned char*>(payload.data()),
                 payload.size(), digest, &dlen);

            std::ostringstream sig;
            for (unsigned i = 0; i < dlen; ++i) {
                sig << std::hex << std::setw(2) << std::setfill('0') 
                    << static_cast<int>(digest[i]);
            }

            // Build request
            json body = {
                {"id", id},
                {"method", method},
                {"api_key", api_key_},
                {"params", params},
                {"nonce", ts},
                {"sig", sig.str()}
            };

            http::request<http::string_body> req{
                http::verb::post,
                full_path,
                11  // HTTP/1.1
            };
            req.set(http::field::host, REST_HOST);
            req.set(http::field::user_agent, USER_AGENT);
            req.set(http::field::content_type, "application/json");
            req.set(http::field::accept, "application/json");
            req.body() = body.dump();
            req.prepare_payload();

            // Send request
            http::write(*socket_, req);

            // Receive response
            beast::flat_buffer buffer;
            http::response<http::string_body> res;
            http::read(*socket_, buffer, res);

            // Process response
            if (res.result() != http::status::ok) {
                throw std::runtime_error("HTTP error: " + std::to_string(res.result_int()));
            }

            json j = json::parse(res.body());
            for (const auto& p : j["result"]["data"]) {
                if (p["instrument_name"] == symbol) {
                    amount = p.value("quantity", 0.0);
                    avg_price = std::abs(p.value("open_pos_cost", 0.0));
                    break;
                }
            }

            // Cleanup
            boost::system::error_code ec;
            socket_->next_layer().shutdown(tcp::socket::shutdown_both, ec);
            socket_->next_layer().close(ec);
            std::this_thread::sleep_for(REQUEST_DELAY);

            return true;

        } catch (const std::exception& e) {
            std::cerr << "[CryptoREST] Error: " << e.what() << "\n";
            if (socket_) socket_->next_layer().close();
            socket_.reset();
            return false;
        }
    }
} // namespace RLTrader
