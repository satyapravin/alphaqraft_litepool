#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace ssl = asio::ssl;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

int main() {
    try {
        // Create IO context
        asio::io_context ioc;

        // Create SSL context
        ssl::context ctx{ssl::context::tlsv12_client};
        ctx.set_default_verify_paths();
        ctx.set_verify_mode(ssl::verify_peer);

        // Create WebSocket stream
        beast::websocket::stream<ssl::stream<tcp::socket>> ws{ioc, ctx};

        // Set SNI hostname
        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), "stream.crypto.com")) {
            throw boost::system::system_error(
                boost::system::error_code(
                    static_cast<int>(::ERR_get_error()),
                    asio::error::get_ssl_category()));
        }

        // Resolve hostname
        tcp::resolver resolver{ioc};
        auto const results = resolver.resolve("stream.crypto.com", "443");

        // Connect TCP layer
        asio::connect(ws.next_layer().next_layer(), results.begin(), results.end());

        // SSL handshake
        ws.next_layer().handshake(ssl::stream_base::client);

        // WebSocket handshake
        ws.set_option(beast::websocket::stream_base::decorator(
            [](beast::websocket::request_type& req) {
                req.set(beast::http::field::host, "stream.crypto.com");
                req.set(beast::http::field::user_agent, "Boost Beast WebSocket");
                req.set(beast::http::field::sec_websocket_protocol, "json");
            }));

        ws.handshake("stream.crypto.com", "/exchange/v1/market");

        // Send subscription message
        json subscribe_msg = {
            {"id", 1},
            {"method", "subscribe"},
            {"params", {
                {"channels", {"book.BTCUSD-PERP.10"}}
            }}
        };

        ws.write(asio::buffer(subscribe_msg.dump()));
        std::cout << "Sent subscription request" << std::endl;

        // Receive messages
        beast::flat_buffer buffer;
        while (true) {
            buffer.clear();
            ws.read(buffer);
            std::cout << "Received: " << beast::buffers_to_string(buffer.data()) << std::endl;
        }

    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
