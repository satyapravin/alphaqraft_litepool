#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <mutex>

namespace RLTrader {
    namespace net = boost::asio;
    namespace ssl = boost::asio::ssl;
    using json = nlohmann::json;

    // Thread safety:
    // - All methods are thread-safe.
    // - connection_mutex_ protects socket_ and connection state.
    class CryptoREST {
    public:
        CryptoREST(const std::string& api_key, const std::string& api_secret);
        ~CryptoREST();

        bool fetch_position(const std::string& symbol, double& amount, double& avgPrice);

    private:
        void do_connect();

        const std::string api_key_;
        const std::string api_secret_;
        net::io_context ioc_;
        ssl::context ssl_ctx_;
        std::unique_ptr<ssl::stream<net::ip::tcp::socket>> socket_;
        std::mutex connection_mutex_;
        std::atomic<bool> running_{false};
    };
}

