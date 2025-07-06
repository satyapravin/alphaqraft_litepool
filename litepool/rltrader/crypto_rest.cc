#include "crypto_rest.h"
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <openssl/hmac.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace RLTrader {
namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
namespace ssl   = net::ssl;
using tcp = net::ip::tcp;
using json = nlohmann::json;

/* Helper: generate Crypto.com REST HMAC sig */
static std::string sign_payload(const std::string& payload, const std::string& secret) {
    unsigned char h[32];
    HMAC(EVP_sha256(),
         secret.data(), secret.size(),
         reinterpret_cast<const unsigned char*>(payload.data()), payload.size(),
         h, nullptr);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for(int i=0;i<32;++i) oss << std::setw(2) << int(h[i]);
    return oss.str();
}

static std::string send_rest(const std::string& body) {
    try {
        net::io_context ioc;
        ssl::context   ctx(ssl::context::tlsv12_client);
        ctx.set_default_verify_paths();
        tcp::resolver  res(ioc);
        //auto const ep = res.resolve("api.crypto.com","443");
        auto const ep = res.resolve("uat-api.3ona.co","443");
        ssl::stream<tcp::socket> stream(ioc,ctx);
        boost::asio::connect(beast::get_lowest_layer(stream), ep);
        stream.handshake(ssl::stream_base::client);

        http::request<http::string_body> req(http::verb::post,"/exchange/v1/private/get-positions",11);
        //req.set(http::field::host,"api.crypto.com");
        req.set(http::field::host,"uat-api.3ona.co");
        req.set(http::field::content_type,"application/json");
        req.body() = body;
        req.prepare_payload();

        http::write(stream,req);
        beast::flat_buffer buf;
        http::response<http::string_body> respo;
        http::read(stream,buf,respo);
        stream.shutdown();
        return respo.body();
    } catch(const std::exception& e){
        std::cerr << "REST error: " << e.what() << std::endl;
        return "";
    }
}

bool CryptoREST::fetch_position(const std::string& symbol,double& amount,double& avgPrice) {
    long nonce = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count();
    std::string method = "private/get-positions";
    std::string id = "1";
    json params = {
        {"instrument_name",symbol}
    };
    std::string payload = method + id + api_key_ + params.dump() + std::to_string(nonce);
    std::string sig = sign_payload(payload, api_secret_);

    json req_body = {
        {"id", id},
        {"method",method},
        {"api_key", api_key_},
        {"sig", sig},
        {"nonce", nonce},
        {"params", params}
    };

    std::string resp = send_rest(req_body.dump());
    if (resp.empty()) return false;

    auto j = json::parse(resp,nullptr,false);
    if (j.is_discarded() || j["code"] != 0) return false;

    const auto& positions = j["result"]["data"];
    for (const auto& p : positions) {
        if (p["instrument_name"] == symbol) {
            amount   = p["quantity"].get<double>();
            avgPrice = p["avg_price"].get<double>();
            return true;
        }
    }
    return false;
}

} // namespace RLTrader
