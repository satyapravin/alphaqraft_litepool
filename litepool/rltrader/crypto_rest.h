// Copyright 2024 Alphaqraft
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
