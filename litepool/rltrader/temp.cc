#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <chrono>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace ws = beast::websocket;
namespace ssl = asio::ssl;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

class WebSocketClient {
public:
    WebSocketClient(asio::io_context& ioc, ssl::context& ctx)
        : ioc_(ioc),
          ctx_(ctx),
          resolver_(ioc),
          ws_(ioc, ctx),
          running_(true),
          connected_(false),
          retry_count_(0) {}

    void start() {
        do_connect();
    }

    void stop() {
        running_ = false;
        if (ws_.is_open()) {
            ws_.async_close(ws::close_code::normal,
                [this](beast::error_code ec) {
                    if (ec) {
                        std::cerr << "Close error: " << ec.message() << '\n';
                    }
                });
        }
    }

private:
void do_connect() {
    if (!running_) return;

    ws_.control_callback(
        [this](ws::frame_type type, beast::string_view payload) {
            if (type == ws::frame_type::pong) {
                std::cout << "Received WebSocket pong: " << payload << '\n';
            }
        });

    auto timeout = std::make_shared<asio::steady_timer>(ioc_);
    timeout->expires_after(std::chrono::seconds(10));
    resolver_.async_resolve("stream.crypto.com", "443",
        [this, timeout](beast::error_code ec, tcp::resolver::results_type results) {
            timeout->cancel();
            if (ec) {
                handle_error("resolve", ec);
                return;
            }
            std::cout << "Resolve completed\n";
            asio::async_connect(ws_.next_layer().next_layer(), results,
                [this, timeout](beast::error_code ec, auto) {
                    timeout->cancel();
                    if (ec) {
                        handle_error("connect", ec);
                        return;
                    }
                    // Set SNI Hostname
                    if (!SSL_set_tlsext_host_name(ws_.next_layer().native_handle(), "stream.crypto.com")) {
                        ec = beast::error_code(static_cast<int>(::ERR_get_error()),
                            asio::error::get_ssl_category());
                        handle_error("SSL_set_tlsext_host_name", ec);
                        return;
                    }

                    ws_.next_layer().async_handshake(ssl::stream_base::client,
                        [this, timeout](beast::error_code ec) {
                            timeout->cancel();
                            if (ec) {
                                handle_error("ssl handshake", ec);
                                return;
                            }
                            // Enhanced WebSocket handshake headers
                            ws_.set_option(ws::stream_base::decorator(
                                [](ws::request_type& req) {
                                    req.set(beast::http::field::host, "stream.crypto.com");
                                    req.set(beast::http::field::user_agent, "Boost Beast WebSocket");
                                    req.set(beast::http::field::accept, "*/*");
                                    req.set(beast::http::field::connection, "upgrade");
                                    req.set(beast::http::field::upgrade, "websocket");
                                    req.set(beast::http::field::sec_websocket_version, "13");
                                    // Add any required API keys here if needed
                                }));
                            ws_.async_handshake("stream.crypto.com", "/v2/market",
                                [this, timeout](beast::error_code ec) {
                                    timeout->cancel();
                                    if (ec) {
                                        handle_error("websocket handshake", ec);
                                        return;
                                    }
                                    std::cout << "WebSocket handshake completed\n";
                                    connected_ = true;
                                    send_subscription();
                                    start_async_read();
                                    do_ping();
                                });
                        });
                });
        });
}
    void send_subscription() {
        json subscribe_msg = {
            {"id", "1"},
            {"method", "subscribe"},
            {"params", {
                {"channels", {"book.BTCUSD-PERP.10"}},
                {"book_subscription_type", "SNAPSHOT"}
            }}
        };
        ws_.async_write(asio::buffer(subscribe_msg.dump()),
            [this](beast::error_code ec, std::size_t) {
                if (ec) {
                    handle_error("write subscription", ec);
                    return;
                }
            });
    }

    void start_async_read() {
        if (!running_ || !connected_) return;

        // Clear the buffer before starting a new read
        buffer_.consume(buffer_.size());

        ws_.async_read(buffer_,
            [this](beast::error_code ec, std::size_t bytes_transferred) {
                if (ec) {
                    if (ec == ws::error::closed) {
                        std::cout << "WebSocket connection closed gracefully\n";
                    } else {
                        std::cerr << "Read error: " << ec.message() << '\n';
                    }
                    handle_error("read", ec);
                    return;
                }

                // Process the received data
                process_received_data(bytes_transferred);

                // Continue reading
                if (running_ && connected_) {
                    start_async_read();
                }
            });
    }

    void process_received_data(std::size_t bytes_transferred) {
        try {
            std::cout << "Received " << bytes_transferred << " bytes\n";
            
            // Parse the JSON message
            auto data = beast::buffers_to_string(buffer_.data());
            json j = json::parse(data, nullptr, false);
            
            if (j.is_discarded()) {
                std::cerr << "Failed to parse message: " << data << '\n';
            } else {
                std::cout << "Received message: " << j.dump(2) << '\n';
                handle_message(j);
            }
            
            // Consume the processed data
            buffer_.consume(bytes_transferred);
        } catch (const std::exception& e) {
            std::cerr << "Error processing data: " << e.what() << '\n';
        }
    }

    void do_ping() {
        if (!running_ || !connected_) return;

        auto timeout = std::make_shared<asio::steady_timer>(ioc_);
        timeout->expires_after(std::chrono::seconds(1));
        timeout->async_wait([this](beast::error_code ec) {
            if (!ec && running_ && connected_) {
                json ping_msg = {{"method", "public/ping"}, {"id", std::to_string(std::time(nullptr))}};
                ws_.async_write(asio::buffer(ping_msg.dump()),
                    [this, ping_msg](beast::error_code ec, std::size_t) {
                        if (ec) {
                            handle_error("write ping", ec);
                            return;
                        }
                        std::cout << "Sent ping: " << ping_msg.dump() << '\n';
                    });
                if (running_) do_ping();
            }
        });
    }

    void handle_message(const json& j) {
        if (!j.contains("method")) {
            std::cerr << "Message missing method: " << j.dump() << '\n';
            return;
        }
        const std::string method = j["method"];
        if (method == "public/heartbeat") {
            if (j.contains("id")) {
                json response = {{"id", j["id"]}, {"method", "public/respond-heartbeat"}};
                ws_.async_write(asio::buffer(response.dump()),
                    [this, response](beast::error_code ec, std::size_t) {
                        if (ec) {
                            handle_error("write heartbeat response", ec);
                            return;
                        }
                        std::cout << "Sent heartbeat response: " << response.dump() << '\n';
                    });
            } else {
                std::cerr << "Heartbeat missing id: " << j.dump() << '\n';
            }
        } else if (method == "public/ping") {
            std::cout << "Received ping response: " << j.dump() << '\n';
        } else if (method == "subscribe" && j.contains("result")) {
            std::cout << "Subscription response: " << j.dump() << '\n';
        } else {
            std::cerr << "Unhandled message: " << j.dump() << '\n';
        }
    }

    void handle_error(const std::string& where, beast::error_code ec) {
        std::cerr << "Error in " << where << ": " << ec.message() << '\n';
        if (ec == asio::error::connection_refused ||
            ec == asio::error::timed_out ||
            ec == ssl::error::stream_truncated ||
            ec == ws::error::closed) {
            connected_ = false;
            auto delay = std::chrono::milliseconds(500);
            std::cout << "Scheduling reconnect after " << delay.count() << "ms\n";
            auto timer = std::make_shared<asio::steady_timer>(ioc_);
            timer->expires_after(delay);
            timer->async_wait([this](beast::error_code ec) {
                if (!ec && running_) {
                    std::cout << "Attempting reconnect\n";
                    retry_count_ = 0;
                    do_connect();
                }
            });
        }
    }

    asio::io_context& ioc_;
    ssl::context& ctx_;
    tcp::resolver resolver_;
    ws::stream<ssl::stream<tcp::socket>> ws_;
    beast::flat_buffer buffer_;
    bool running_;
    bool connected_;
    int retry_count_;
};

int main() {
    try {
        asio::io_context ioc;
        ssl::context ctx{ssl::context::tlsv12_client};
        ctx.set_default_verify_paths();
        ctx.set_verify_mode(ssl::verify_peer);

        WebSocketClient client(ioc, ctx);
        client.start();

        ioc.run();
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
