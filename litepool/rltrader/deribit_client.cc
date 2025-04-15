// deribit_client.cc
#include "deribit_client.h"
#include <iostream>
#include <utility>
#include <boost/asio/connect.hpp>
#include <boost/asio/post.hpp>

namespace RLTrader {

DeribitClient::DeribitClient(std::string api_key,
                           std::string api_secret,
                           std::string symbol,
                           std::string hedge_symbol)
    : api_key_(std::move(api_key)),
      api_secret_(std::move(api_secret)),
      symbol_(std::move(symbol)),
      hedge_symbol_(hedge_symbol.empty() ? symbol_ : std::move(hedge_symbol)) {}

DeribitClient::~DeribitClient() {
    stop();
}

void DeribitClient::start() {
    if (running_.exchange(true)) return;
    
    market_ioc_ = std::make_unique<net::io_context>();
    trading_ioc_ = std::make_unique<net::io_context>();
    ctx_ = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
    
    ctx_->set_verify_mode(ssl::verify_peer);
    ctx_->set_default_verify_paths();

    setup_connections();
    
    market_thread_ = std::make_unique<std::thread>([this] { market_ioc_->run(); });
    trading_thread_ = std::make_unique<std::thread>([this] { trading_ioc_->run(); });
}

void DeribitClient::stop() {
    if (!running_.exchange(false)) return;

    // Stop IO contexts
    if (market_ioc_) market_ioc_->stop();
    if (trading_ioc_) trading_ioc_->stop();
    
    // Join threads
    if (market_thread_ && market_thread_->joinable()) market_thread_->join();
    if (trading_thread_ && trading_thread_->joinable()) trading_thread_->join();

    // Reset resources
    market_ws_.reset();
    trading_ws_.reset();
    market_ioc_.reset();
    trading_ioc_.reset();
    ctx_.reset();
}

void DeribitClient::setup_connections() {
    setup_market_connection();
    setup_trading_connection();
}

void DeribitClient::setup_market_connection() {
    market_ws_ = std::make_unique<websocket_stream>(*market_ioc_, *ctx_);
    do_market_connect();
}

void DeribitClient::setup_trading_connection() {
    trading_ws_ = std::make_unique<websocket_stream>(*trading_ioc_, *ctx_);
    do_trading_connect();
}

void DeribitClient::do_market_connect() {
    constexpr auto host = "www.deribit.com";
    constexpr auto port = "443";

    tcp::resolver resolver(*market_ioc_);
    resolver.async_resolve(host, port, 
        [this](const boost::system::error_code& ec, tcp::resolver::results_type results) {
            if (ec) return handle_error("Market Resolve", ec);
            
            // Get the lowest layer (TCP socket)
            auto& lowest_layer = beast::get_lowest_layer(*market_ws_);
            
            // Connect to first endpoint
            lowest_layer.async_connect(
                *results, // Dereference to get first endpoint
                [this](const boost::system::error_code& ec, const tcp::endpoint&) {
                    if (ec) return handle_error("Market Connect", ec);
                    
                    // SSL handshake
                    market_ws_->next_layer().async_handshake(
                        ssl::stream_base::client,
                        [this](const boost::system::error_code& ec) {
                            if (ec) return handle_error("Market SSL", ec);
                            
                            // WebSocket handshake
                            market_ws_->async_handshake(
                                host, "/ws/api/v2",
                                [this](const boost::system::error_code& ec) {
                                    if (ec) return handle_error("Market WS", ec);
                                    
                                    market_connected_ = true;
                                    subscribe_market_data();
                                    do_market_read();
                                    start_heartbeat();
                                });
                        });
                });
        });
}

void DeribitClient::do_trading_connect() {
    constexpr auto host = "www.deribit.com";
    constexpr auto port = "443";

    tcp::resolver resolver(*trading_ioc_);
    resolver.async_resolve(host, port,
        [this](const boost::system::error_code& ec, tcp::resolver::results_type results) {
            if (ec) return handle_error("Trading Resolve", ec);
            
            // Get the lowest layer (TCP socket)
            auto& lowest_layer = beast::get_lowest_layer(*trading_ws_);
            
            // Connect to first endpoint
            lowest_layer.async_connect(
                *results, // Dereference to get first endpoint
                [this](const boost::system::error_code& ec, const tcp::endpoint&) {
                    if (ec) return handle_error("Trading Connect", ec);
                    
                    // SSL handshake
                    trading_ws_->next_layer().async_handshake(
                        ssl::stream_base::client,
                        [this](const boost::system::error_code& ec) {
                            if (ec) return handle_error("Trading SSL", ec);
                            
                            // WebSocket handshake
                            trading_ws_->async_handshake(
                                host, "/ws/api/v2",
                                [this](const boost::system::error_code& ec) {
                                    if (ec) return handle_error("Trading WS", ec);
                                    
                                    trading_connected_ = true;
                                    authenticate();
                                    do_trading_read();
                                    start_heartbeat();
                                });
                        });
                });
        });
}

void DeribitClient::start_heartbeat() {
    const json heartbeat_msg = {
        {"jsonrpc", "2.0"},
        {"method", "public/heartbeat"},
        {"params", {{"type", "test_request"}}},
        {"id", 999}
    };

    if (market_connected_) {
        market_timer_ = std::make_unique<net::steady_timer>(*market_ioc_);
        
        // Define the schedule_heartbeat function first
        std::function<void()> schedule_heartbeat = [this, heartbeat_msg, &schedule_heartbeat]() {
            market_timer_->expires_after(std::chrono::seconds(10));
            market_timer_->async_wait([this, heartbeat_msg, &schedule_heartbeat](const beast::error_code& ec) {
                if (!ec && running_) {
                    send_market_message(json(heartbeat_msg));
                    schedule_heartbeat();
                }
            });
        };
        
        schedule_heartbeat();
    }

    if (trading_connected_) {
        trading_timer_ = std::make_unique<net::steady_timer>(*trading_ioc_);
        
        // Define the schedule_heartbeat function first
        std::function<void()> schedule_heartbeat = [this, heartbeat_msg, &schedule_heartbeat]() {
            trading_timer_->expires_after(std::chrono::seconds(10));
            trading_timer_->async_wait([this, heartbeat_msg, &schedule_heartbeat](const beast::error_code& ec) {
                if (!ec && running_) {
                    send_trading_message(json(heartbeat_msg));
                    schedule_heartbeat();
                }
            });
        };
        
        schedule_heartbeat();
    }
}


void DeribitClient::authenticate() {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", "public/auth"},
        {"params", {
            {"grant_type", "client_credentials"},
            {"client_id", api_key_},
            {"client_secret", api_secret_}
        }},
        {"id", 0}
    });
}

void DeribitClient::subscribe_market_data() {
    send_market_message({
        {"jsonrpc", "2.0"},
        {"method", "public/subscribe"},
        {"params", {
            {"channels", {
                "book." + symbol_ + ".none.20.100ms"
            }}
        }},
        {"id", 1}
    });
}

void DeribitClient::subscribe_private_data() {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", "private/subscribe"},
        {"params", {
            {"channels", {
                "user.trades." + symbol_ + ".raw",
                "user.orders." + symbol_ + ".raw",
                "user.portfolio." + symbol_
            }}
        }},
        {"id", 2}
    });
}

// Trading operations implementation
void DeribitClient::place_order(const std::string& side, 
                               double price, 
                               double size,
                               const std::string& label,
                               bool is_hedge,
                               const std::string& type) {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", side == "buy" ? "private/buy" : "private/sell"},
        {"params", {
            {"instrument_name", is_hedge ? hedge_symbol_ : symbol_},
            {"amount", size},
            {"type", type},
            {"price", price},
            {"label", label},
            {"post_only", type == "limit"}
        }},
        {"id", 3}
    });
}

void DeribitClient::cancel_order(const std::string& order_id) {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", "private/cancel"},
        {"params", {{"order_id", order_id}}},
        {"id", 4}
    });
}

void DeribitClient::cancel_all_by_label(const std::string& label) {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", "private/cancel_by_label"},
        {"params", {{"label", label}}},
        {"id", 7}
    });
}

void DeribitClient::cancel_all_orders() {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", "private/cancel_all"},
        {"params", {{"instrument_name", symbol_}}},
        {"id", 5}
    });
}

void DeribitClient::get_position() {
    send_trading_message({
        {"jsonrpc", "2.0"},
        {"method", "private/get_position"},
        {"params", {{"instrument_name", symbol_}}},
        {"id", 6}
    });
}

// Callback setters
void DeribitClient::set_OrderBook_cb(std::function<void(const json&)> cb) {
    OrderBook_cb_ = std::move(cb);
}

void DeribitClient::set_private_trade_cb(std::function<void(const json&)> cb) {
    private_trade_cb_ = std::move(cb);
}

void DeribitClient::set_position_cb(std::function<void(const json&)> cb) {
    position_cb_ = std::move(cb);
}

void DeribitClient::set_order_cb(std::function<void(const json&)> cb) {
    order_cb_ = std::move(cb);
}

// Message handling
void DeribitClient::do_market_read() {
    market_ws_->async_read(
        market_buffer_,
        [this](const beast::error_code& ec, std::size_t) {
            if (ec) return handle_error("Market Read", ec);
            
            try {
                auto msg = json::parse(beast::buffers_to_string(market_buffer_.data()));
                market_buffer_.consume(market_buffer_.size());
                handle_market_message(msg);
            } catch (const std::exception& e) {
                std::cerr << "Market parse error: " << e.what() << std::endl;
            }
            
            if (running_) do_market_read();
        });
}

void DeribitClient::do_trading_read() {
    trading_ws_->async_read(
        trading_buffer_,
        [this](const beast::error_code& ec, std::size_t) {
            if (ec) return handle_error("Trading Read", ec);
            
            try {
                auto msg = json::parse(beast::buffers_to_string(trading_buffer_.data()));
                trading_buffer_.consume(trading_buffer_.size());
                handle_trading_message(msg);
            } catch (const std::exception& e) {
                std::cerr << "Trading parse error: " << e.what() << std::endl;
            }
            
            if (running_) do_trading_read();
        });
}

void DeribitClient::handle_market_message(const json& msg) {
    if (msg.contains("method") && msg["method"] == "subscription") {
        const auto& channel = msg["params"]["channel"].get<std::string>();
        if (channel.find("book.") == 0 && OrderBook_cb_) {
            OrderBook_cb_(msg["params"]["data"]);
        }
    }
}

void DeribitClient::handle_trading_message(const json& msg) {
    if (msg.contains("method") && msg["method"] == "subscription") {
        const auto& channel = msg["params"]["channel"].get<std::string>();
        if (channel.find("user.trades.") == 0 && private_trade_cb_) {
            private_trade_cb_(msg["params"]["data"]);
        }
        else if (channel.find("user.orders.") == 0 && order_cb_) {
            order_cb_(msg["params"]["data"]);
        }
        else if (channel.find("user.portfolio.") == 0 && position_cb_) {
            position_cb_(msg["params"]["data"]);
        }
    }
    else if (msg.contains("id") && msg["id"] == 0 && 
             msg.contains("result") && msg["result"]["token_type"] == "bearer") {
        subscribe_private_data();
    }
}

void DeribitClient::send_market_message(json&& msg) {
    if (!market_out_queue_.push(std::move(msg))) {
        std::cerr << "Market output queue full!" << std::endl;
        return;
    }
    net::post(*market_ioc_, [this] { write_next_market_message(); });
}

void DeribitClient::send_trading_message(json&& msg) {
    if (!trading_out_queue_.push(std::move(msg))) {
        std::cerr << "Trading output queue full!" << std::endl;
        return;
    }
    net::post(*trading_ioc_, [this] { write_next_trading_message(); });
}

void DeribitClient::write_next_market_message() {
    if (!market_connected_ || market_writing_) return;
    
    json msg;
    if (market_out_queue_.pop(msg)) {
        market_writing_ = true;
        const auto msg_str = msg.dump();
        market_ws_->async_write(
            net::buffer(msg_str),
            [this](const beast::error_code& ec, std::size_t) {
                market_writing_ = false;
                if (ec) {
                    handle_error("Market Write", ec);
                } else if (!market_out_queue_.empty()) {
                    write_next_market_message();
                }
            });
    }
}

void DeribitClient::write_next_trading_message() {
    if (!trading_connected_ || trading_writing_) return;
    
    json msg;
    if (trading_out_queue_.pop(msg)) {
        trading_writing_ = true;
        const auto msg_str = msg.dump();
        trading_ws_->async_write(
            net::buffer(msg_str),
            [this](const beast::error_code& ec, std::size_t) {
                trading_writing_ = false;
                if (ec) {
                    handle_error("Trading Write", ec);
                } else if (!trading_out_queue_.empty()) {
                    write_next_trading_message();
                }
            });
    }
}

void DeribitClient::handle_error(const std::string& context, const beast::error_code& ec) {
    std::cerr << context << " error: " << ec.message() << std::endl;
    
    if (context.find("Market") != std::string::npos) {
        market_connected_ = false;
        if (running_) setup_market_connection();
    } 
    else if (context.find("Trading") != std::string::npos) {
        trading_connected_ = false;
        if (running_) setup_trading_connection();
    }
}

} // namespace RLTrader
