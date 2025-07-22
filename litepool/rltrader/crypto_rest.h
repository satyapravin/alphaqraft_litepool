#pragma once

#include <string>
#include <mutex>
#include <memory>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>

namespace RLTrader
{
    namespace net = boost::asio;
    namespace ssl = boost::asio::ssl;
    using json = nlohmann::json;
    
    class CryptoREST
    {
    public:
        CryptoREST(const std::string& api_key,
                   const std::string& api_secret);
        ~CryptoREST();

        bool fetch_position(const std::string& symbol,
                           double& amount,
                           double& avg_price);

    private:
        void do_connect();
        bool ensure_connection();
        static std::string build_payload(const std::string& method, 
                                       int id, 
                                       const std::string& api_key,
                                       const json& params, 
                                       long nonce);
        const std::string api_key_;
        const std::string api_secret_;

        net::io_context ioc_;
        ssl::context ssl_ctx_{ssl::context::tlsv12_client};

        std::unique_ptr<ssl::stream<net::ip::tcp::socket>> socket_;
        std::mutex connection_mutex_;
    };
}
