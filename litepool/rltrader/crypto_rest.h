#pragma once

#include <atomic>
#include <mutex>
#include <string>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <nlohmann/json.hpp>

namespace RLTrader
{
    namespace net = boost::asio;
    namespace ssl = boost::asio::ssl;
    using      json = nlohmann::json;

    // Thread-safe HTTPS helper (Crypto.com REST V1)
    class CryptoREST
    {
    public:
        CryptoREST(const std::string& api_key,
                   const std::string& api_secret);
        ~CryptoREST();

        // returns true on success – amount / avgPrice are always written
        bool fetch_position(const std::string& symbol,
                            double&            amount,
                            double&            avgPrice);

    private:
        void do_connect();

        const std::string api_key_;
        const std::string api_secret_;

        net::io_context ioc_;
        ssl::context    ssl_ctx_{ssl::context::tlsv12_client};

        std::unique_ptr<ssl::stream<net::ip::tcp::socket>> socket_;
        std::mutex                                         connection_mutex_;
        std::atomic<bool>                                  running_{true};
    };
} // namespace RLTrader
