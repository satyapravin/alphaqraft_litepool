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

std::string send_get_request(const std::string& host, const std::string& target, const std::string& token) {
    try {
        net::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        tcp::resolver resolver(ioc);
        auto const results = resolver.resolve(host, "443");

        ssl::stream<tcp::socket> stream(ioc, ctx);
        
        if(!SSL_set_tlsext_host_name(stream.native_handle(), "www.deribit.com")) {
            throw beast::system_error(
                beast::error_code(
                    static_cast<int>(ERR_get_error()),
                    net::error::get_ssl_category()
                ),
                "Failed to set SNI Hostname"
            );
        }
        connect(stream.next_layer(), results.begin(), results.end());
        stream.handshake(ssl::stream_base::client);

        http::request<http::empty_body> req(http::verb::get, target, 11);
        req.set(http::field::host, host);
        req.set(http::field::content_type, "application/json");
        req.set(http::field::authorization, "Bearer " + token);

        http::write(stream, req);
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);

        // Handle SSL shutdown safely
        boost::system::error_code ec;
        stream.shutdown(ec);
        if (ec == boost::asio::error::eof || ec == boost::asio::ssl::error::stream_truncated) {
            ec.clear();  // Ignore expected shutdown errors
        } else if (ec) {
            std::cerr << "SSL Shutdown Error: " << ec.message() << std::endl;
        }

        return res.body();
    } catch (const std::exception& e) {
        std::cerr << "Request failed: " << e.what() << std::endl;
        return "";
    }
}

std::string get_bearer_token(const std::string& client_id, 
                           const std::string& client_secret) {
    std::string auth_body = R"({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "public/auth",
        "params": {
            "grant_type": "client_credentials",
            "client_id": ")" + client_id + R"(",
            "client_secret": ")" + client_secret + R"("
        }
    })";

    std::string response = send_get_request("www.deribit.com", "/api/v2/public/auth", auth_body);

    json jsonResponse = json::parse(response);
    if (jsonResponse.contains("result") && jsonResponse["result"].contains("access_token")) {
        return jsonResponse["result"]["access_token"];
    } else {
        std::cerr << "Authentication failed: " << response << std::endl;
        return "";
    }
}

void DeribitREST::fetch_position(const std::string& symbol, double& amount, double& price) {
    try {
        auto bearer_token = get_bearer_token(api_key, api_secret);
        if (bearer_token.empty()) {
            std::cerr << "Failed to obtain bearer token." << std::endl;
            return;
        }

        std::string query = "/api/v2/private/get_position?instrument_name=" + symbol;
        
        net::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        ctx.set_verify_mode(ssl::verify_peer);
        ctx.set_default_verify_paths();
        
        tcp::resolver resolver(ioc);
        beast::ssl_stream<beast::tcp_stream> stream(ioc, ctx);
        
        if(!SSL_set_tlsext_host_name(stream.native_handle(), "www.deribit.com")) {
            throw beast::system_error(
                beast::error_code(
                    static_cast<int>(ERR_get_error()),
                    net::error::get_ssl_category()
                ),
                "Failed to set SNI Hostname"
            );
        }
        
        auto const results = resolver.resolve("www.deribit.com", "443");
        beast::get_lowest_layer(stream).connect(results);
        stream.handshake(ssl::stream_base::client);
        
        // Set up GET request
        http::request<http::string_body> req{http::verb::get, query, 11};
        req.set(http::field::host, "www.deribit.com");
        req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        req.set(http::field::content_type, "application/json");
        
        std::string auth = "Bearer " + bearer_token;
        req.set(http::field::authorization, auth);
        
        // Debug output
        std::cout << "Authorization: " << auth << std::endl;
        
        // Send the request
        http::write(stream, req);
        
        // Receive the response
        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);
        
        std::cout << "Response status: " << res.result_int() << std::endl;
        std::cout << "Response body: " << res.body() << std::endl;
        
        if (res.result() == http::status::ok) {
            auto j = json::parse(res.body());
            
            if (j.contains("result") && !j["result"].is_null()) {
                const auto& result = j["result"];
                auto size = result["size"].get<double>();
                price = result["average_price"].get<double>();
                amount = result["direction"].get<std::string>() == "buy" ? size : -std::abs(size);
            } else if (j.contains("error")) {
                throw std::runtime_error("API error: " + 
                    j["error"]["message"].get<std::string>());
            }
        } else {
            throw std::runtime_error("HTTP error: " + 
                std::to_string(res.result_int()) + 
                " Body: " + res.body());
        }
        
        // Gracefully close the stream
        beast::error_code ec;
        stream.shutdown(ec);
        if(ec == net::error::eof) {
            ec = {};
        }
        if(ec) {
            throw beast::system_error{ec};
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting position: " << e.what() << std::endl;
    }
}
