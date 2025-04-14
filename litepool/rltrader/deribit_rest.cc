#include "deribit_rest.h"

#include <boost/beast/core.hpp>
#include <boost/asio.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ssl/stream.hpp>

#include <openssl/hmac.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;
using json = nlohmann::json;

using namespace RLTrader;

static constexpr char DERIBIT_HOST[] = "www.deribit.com";

std::string url_encode(const std::string &value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : value) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else {
            escaped << '%' << std::setw(2) << int((unsigned char)c);
        }
    }

    return escaped.str();
}

template<typename RequestBody, typename ResponseBody>
std::string send_request(const std::string& target, 
                         http::verb method,
                         const std::function<void(http::request<RequestBody>&)>& request_setup) {
    try {
        net::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        
        // Load certificates but don't strictly verify
        ctx.set_default_verify_paths();
        ctx.set_verify_mode(ssl::verify_none);
        
        tcp::resolver resolver(ioc);
        auto const results = resolver.resolve(DERIBIT_HOST, "443");
        
        ssl::stream<tcp::socket> stream(ioc, ctx);
        
        // Set SNI before connecting
        if (!SSL_set_tlsext_host_name(stream.native_handle(), DERIBIT_HOST)) {
            throw beast::system_error(beast::error_code(static_cast<int>(ERR_get_error()), 
                                     net::error::get_ssl_category()),
                                     "Failed to set SNI Hostname");
        }
        
        // Connect to the server
        boost::asio::connect(beast::get_lowest_layer(stream), results);
        
        // Perform handshake with better error handling
        try {
            stream.handshake(ssl::stream_base::client);
        } catch (const beast::system_error& e) {
            std::cerr << "SSL handshake failed: " << e.what() << std::endl;
            return "";
        }
        
        // Set up the request
        http::request<RequestBody> req(method, target, 11);
        req.set(http::field::host, DERIBIT_HOST);
        req.set(http::field::user_agent, "Boost Beast");
        
        // Apply custom request setup
        request_setup(req);
        
        // Send the request
        http::write(stream, req);
        
        // Read the response
        beast::flat_buffer buffer;
        http::response<ResponseBody> res;
        
        try {
            http::read(stream, buffer, res);
        } catch (const beast::system_error& e) {
            if (e.code() == http::error::end_of_stream) {
                std::cerr << "Server closed connection" << std::endl;
            } else {
                std::cerr << "Read error: " << e.what() << std::endl;
            }
            
            // Even with an error, we might have a partial response
            if (res.body().empty()) {
                return "";
            }
        }
        
        // Graceful shutdown with error handling
        beast::error_code ec;
        stream.shutdown(ec);
        
        // Ignore common shutdown errors
        if (ec == net::error::eof || 
            ec == ssl::error::stream_truncated ||
            ec == net::ssl::error::stream_errors::stream_truncated) {
            // These are expected errors when shutting down SSL connections
            ec = {};
        }
        
        if (ec) {
            std::cerr << "Shutdown error: " << ec.message() << std::endl;
            // Continue anyway, we already have the response
        }
        
        return res.body();
    } catch (const std::exception& e) {
        std::cerr << "Request failed: " << e.what() << std::endl;
        return "";
    }
}

std::string send_get_request(const std::string& target, const std::string& token) {
    return send_request<http::empty_body, http::string_body>(
        target, 
        http::verb::get,
        [&token](auto& req) {
            req.set(http::field::authorization, "Bearer " + token);
        }
    );
}

std::string send_post_request(const std::string& target, const std::string& body) {
    return send_request<http::string_body, http::string_body>(
        target, 
        http::verb::post,
        [&body](auto& req) {
            req.set(http::field::content_type, "application/json");
            req.body() = body;
            req.prepare_payload();
        }
    );
}

std::string get_bearer_token(const std::string& client_id, const std::string& client_secret) {
    try {
        json auth_json = {
            {"jsonrpc", "2.0"}, 
            {"id", 1}, 
            {"method", "public/auth"},
            {"params", {
                {"grant_type", "client_credentials"},
                {"client_id", client_id},
                {"client_secret", client_secret}
            }}
        };

        std::string response = send_post_request("/api/v2/public/auth", auth_json.dump());

        if (response.empty()) {
            throw std::runtime_error("Empty response received from server");
        }

        auto jsonResponse = json::parse(response, nullptr, false);
        if (jsonResponse.is_discarded() || !jsonResponse.contains("result") || !jsonResponse["result"].contains("access_token")) {
            throw std::runtime_error("Authentication failed: " + response);
        }

        return jsonResponse["result"]["access_token"].get<std::string>();
    } catch (const std::exception& e) {
        std::cerr << "Failed to obtain bearer token: " << e.what() << std::endl;
        return "";
    }
}

void DeribitREST::fetch_position(const std::string& symbol, double& amount, double& price) {
    try {
        auto bearer_token = get_bearer_token(api_key, api_secret);
        if (bearer_token.empty()) {
            throw std::runtime_error("Failed to obtain bearer token");
        }
       
        std::string query = "/api/v2/private/get_position?instrument_name=" + symbol;
        std::string response = send_get_request(query, bearer_token);

        if (response.empty()) {
            throw std::runtime_error("Empty response from server");
        }

        auto j = json::parse(response, nullptr, false);
        if (j.is_discarded()) {
            throw std::runtime_error("Invalid JSON response");
        }

        if (j.contains("error")) {
            throw std::runtime_error("API error: " + j["error"]["message"].get<std::string>());
        }

        if (j.contains("result") && !j["result"].is_null()) {
            const auto& result = j["result"];
            if (result.contains("size") && result.contains("average_price") && result.contains("direction")) {
                auto size = result["size"].get<double>();
                price = result["average_price"].get<double>();
                amount = result["direction"].get<std::string>() == "buy" ? size : -std::abs(size);
            } else {
                throw std::runtime_error("Missing expected fields in API response");
            }
        } else {
            // No position exists for this instrument
            amount = 0.0;
            price = 0.0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error getting position: " << e.what() << std::endl;
        // Initialize with default values on error
        amount = 0.0;
        price = 0.0;
    }
}
