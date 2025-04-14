#include "deribit_client.h"

#include <utility>
#include <iostream>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/ssl.hpp>
#include <nlohmann/json.hpp>

using namespace RLTrader;

char to_lower(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

bool starts_with(const std::string& str, const std::string& prefix) {
    if (str.size() < prefix.size()) {
        return false;
    }

    return std::equal(
        prefix.begin(), prefix.end(),
        str.begin(),
        [](char a, char b) {
            return to_lower(a) == to_lower(b);
        }
    );
}

DeribitClient::DeribitClient(const std::string& api_key,
                             const std::string& api_secret,
                             const std::string& symbol,
			     const std::string& hedge_symbol)
    : api_key_(api_key)
    , api_secret_(api_secret)
    , symbol_(symbol) 
    , hedge_symbol_(hedge_symbol) {
}

DeribitClient::~DeribitClient() {
    stop();
}

void DeribitClient::start() {
    if (should_run_) return;
    
    should_run_ = true;
    ioc_ = std::make_unique<net::io_context>();
    ctx_ = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
    
    ctx_->set_verify_mode(ssl::verify_peer);
    ctx_->set_default_verify_paths();

    setup_connections();
    
    io_thread_ = std::make_unique<std::thread>([this] { run_io_context(); });
}

void DeribitClient::stop() {
    if (!should_run_) return;
    
    should_run_ = false;

    // Clear message queues
    {
        std::lock_guard<std::mutex> market_lock(market_write_mutex_);
        std::queue<json>().swap(market_message_queue_);
    }
    {
        std::lock_guard<std::mutex> trading_lock(trading_write_mutex_);
        std::queue<json>().swap(trading_message_queue_);
    }

    if (trading_connected_) {
        cancel_all_orders();
    }

    if (ioc_) {
        ioc_->stop();
    }
    
    if (io_thread_ && io_thread_->joinable()) {
        io_thread_->join();
    }

    {
        std::lock_guard<std::mutex> market_lock(market_write_mutex_);
        market_ws_.reset();
    }

    {
        std::lock_guard<std::mutex> trading_lock(trading_write_mutex_);
        trading_ws_.reset();
    }

    ioc_.reset();
    ctx_.reset();
}

void DeribitClient::run_io_context() {
    while (should_run_) {
        try {
            ioc_->run();
            if (!should_run_) break;
            
            ioc_->restart();
            
            if (!market_connected_ || !trading_connected_) {
                setup_connections();
            }
        }
        catch (const std::exception& e) {
            std::cerr << "IO context error: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void DeribitClient::setup_connections() {
    setup_market_connection();
    setup_trading_connection();
}

void DeribitClient::setup_market_connection() {
    market_ws_ = std::make_unique<websocket::stream<beast::ssl_stream<tcp::socket>>>(*ioc_, *ctx_);
    do_market_connect();
}

void DeribitClient::setup_trading_connection() {
    trading_ws_ = std::make_unique<websocket::stream<beast::ssl_stream<tcp::socket>>>(*ioc_, *ctx_);
    do_trading_connect();
}

void DeribitClient::do_market_connect() {
    auto const host = "www.deribit.com";
    auto const port = "443";

    tcp::resolver resolver(*ioc_);
    auto const results = resolver.resolve(host, port);

    // Get the lowest layer (TCP socket) for connection
    auto& lowest_layer = beast::get_lowest_layer(*market_ws_);

    net::async_connect(
        lowest_layer,
        results,
        [this](const beast::error_code& ec, const tcp::endpoint&) {
            if (ec) {
                handle_error("Market Connect", ec);
                return;
            }

            // SSL Handshake
            market_ws_->next_layer().async_handshake(
                ssl::stream_base::client,
                [this](const beast::error_code& ec) {
                    if (ec) {
                        handle_error("Market SSL", ec);
                        return;
                    }

                    // WebSocket Handshake
                    market_ws_->async_handshake(
                        "www.deribit.com",
                        "/ws/api/v2",
                        [this](const beast::error_code& ec) {
                            if (ec) {
                                handle_error("Market WS", ec);
                                return;
                            }

                            market_connected_ = true;
                            subscribe_market_data();
                            do_market_read();
                        });
                });
        });
}

void DeribitClient::do_trading_connect() {
    auto const host = "www.deribit.com";
    auto const port = "443";

    tcp::resolver resolver(*ioc_);
    auto const results = resolver.resolve(host, port);

    // Get the lowest layer (TCP socket) for connection
    auto& lowest_layer = beast::get_lowest_layer(*trading_ws_);

    net::async_connect(
        lowest_layer,
        results,
        [this](const beast::error_code& ec, const tcp::endpoint&) {
            if (ec) {
                handle_error("Trading Connect", ec);
                return;
            }

            // SSL Handshake
            trading_ws_->next_layer().async_handshake(
                ssl::stream_base::client,
                [this](const beast::error_code& ec) {
                    if (ec) {
                        handle_error("Trading SSL", ec);
                        return;
                    }

                    // WebSocket Handshake
                    trading_ws_->async_handshake(
                        "www.deribit.com",
                        "/ws/api/v2",
                        [this](const beast::error_code& ec) {
                            if (ec) {
                                handle_error("Trading WS", ec);
                                return;
                            }
                            
                            trading_connected_ = true;
                            authenticate();
                            do_trading_read();
                        });
                });
        });
}

void DeribitClient::authenticate() {
    json auth_msg = {
        {"jsonrpc", "2.0"},
        {"method", "public/auth"},
        {"params", {
            {"grant_type", "client_credentials"},
            {"client_id", api_key_},
            {"client_secret", api_secret_}
        }},
        {"id", 0}
    };
    
    send_trading_message(auth_msg);
}

void DeribitClient::subscribe_market_data() {
    if (!market_connected_) return;
    
    json sub_msg = {
        {"jsonrpc", "2.0"},
        {"method", "public/subscribe"},
        {"params", {
            {"channels", {
                "book." + symbol_ + ".none.20.100ms"
            }}
        }},
        {"id", 1}
    };
    
    send_market_message(sub_msg);
}

void DeribitClient::subscribe_private_data() {
    if (!trading_connected_) return;
    
    json sub_msg = {
        {"jsonrpc", "2.0"},
        {"method", "private/subscribe"},
        {"params", {
            {"channels", {
                "user.trades." + symbol_ + ".100ms",
                "user.orders." + symbol_ + ".raw",
                "user.portfolio." + symbol_
            }}
        }},
        {"id", 2}
    };
    
    send_trading_message(sub_msg);
}

void DeribitClient::place_order(const std::string& side, 
                              double price, 
                              double size,
                              const std::string& label,
			      bool is_hedge,
                              const std::string& type) {
    if (!trading_connected_) {
	    std::cout << "disconnected deribit markets!" << std::endl;
	    return;
    }
    
    bool post_only = !(type == "market"); 
    json order_msg = {
        {"jsonrpc", "2.0"},
        {"method", side == "buy" ? "private/buy" : "private/sell"},
        {"params", {
            {"instrument_name", is_hedge ? hedge_symbol_ : symbol_},
            {"amount", size},
            {"type", type},
            {"price", price},
            {"label", label},
            {"post_only", post_only}
        }},
        {"id", 3}
    };
    
    send_trading_message(order_msg);
}

void DeribitClient::cancel_order(const std::string& order_id) {
    if (!trading_connected_) return;
    
    json cancel_msg = {
        {"jsonrpc", "2.0"},
        {"method", "private/cancel"},
        {"params", {
            {"order_id", order_id}
        }},
        {"id", 4}
    };
    
    send_trading_message(cancel_msg);
}

void DeribitClient::cancel_all_by_label(const std::string& label) {
    if (!trading_connected_) return;

    json cancel_by_label_msg = {
        {"jsonrpc", "2.0"},
        {"method", "private/cancel_by_label"},
        {"params", {
                {"label", label}
        }},
        {"id", 7}
    };

    send_trading_message(cancel_by_label_msg);
}
void DeribitClient::cancel_all_orders() {
    if (!trading_connected_) return;
    
    json cancel_all_msg = {
        {"jsonrpc", "2.0"},
        {"method", "private/cancel_all"},
        {"params", {
            {"instrument_name", symbol_}
        }},
        {"id", 5}
    };
    
    send_trading_message(cancel_all_msg);
}

void DeribitClient::get_position() {
    if (!trading_connected_) return;
    
    json pos_msg = {
        {"jsonrpc", "2.0"},
        {"method", "private/get_position"},
        {"params", {
            {"instrument_name", symbol_}
        }},
        {"id", 6}
    };
    
    send_trading_message(pos_msg);
}

void DeribitClient::set_OrderBook_cb(std::function<void(const json&)> OrderBook_cb) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    OrderBook_callback_ = std::move(OrderBook_cb);
}

void DeribitClient::set_private_trade_cb(std::function<void(const json&)> private_trade_cb) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    private_trade_callback_ = std::move(private_trade_cb);
}

void DeribitClient::set_position_cb(std::function<void(const json&)> position_cb) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    position_callback_ = std::move(position_cb);
}

void DeribitClient::set_order_cb(std::function<void(const json&)> order_cb) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    order_callback_ = std::move(order_cb);
}

void DeribitClient::do_market_read() {
    market_ws_->async_read(
        market_buffer_,
        [this](const beast::error_code& ec, std::size_t) {
            if (ec) {
                handle_error("Market Read", ec);
                return;
            }
            
            std::string msg = beast::buffers_to_string(market_buffer_.data());
            market_buffer_.consume(market_buffer_.size());
            
            try {
                auto j = json::parse(msg);
                handle_market_message(j);
            } catch (const std::exception& e) {
                std::cerr << "Market parse error: " << e.what() << std::endl;
            }
            
            if (should_run_) {
                do_market_read();
            }
        });
}

void DeribitClient::do_trading_read() {
    trading_ws_->async_read(
        trading_buffer_,
        [this](const beast::error_code& ec, std::size_t) {
            if (ec) {
                handle_error("Trading Read", ec);
                return;
            }
            
            std::string msg = beast::buffers_to_string(trading_buffer_.data());
            trading_buffer_.consume(trading_buffer_.size());
            
            try {
                auto j = json::parse(msg);
                handle_trading_message(j);
            } catch (const std::exception& e) {
                std::cerr << "Trading parse error: " << e.what() << std::endl;
            }
            
            if (should_run_) {
                do_trading_read();
            }
        });
}

void DeribitClient::handle_market_message(const json& msg) {
    if (msg.contains("method") && msg["method"] == "subscription") {
        const std::string& channel = msg["params"]["channel"];
        if (starts_with(channel, "book.")) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (OrderBook_callback_) {
                OrderBook_callback_(msg["params"]["data"]);
            }
        }
    }
}

void DeribitClient::handle_trading_message(const json& msg) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    
    if (msg.contains("method") && msg["method"] == "subscription") {
        if (const std::string& channel = msg["params"]["channel"]; starts_with(channel, "user.trades.")) {
            if (private_trade_callback_) {
                private_trade_callback_(msg["params"]["data"]);
            }
        }
        else if (starts_with(channel, "user.orders.")) {
            if (order_callback_) {
                order_callback_(msg["params"]["data"]);
            }
        }
        else if (starts_with(channel, "user.portfolio.")) {
            if (position_callback_) {
                position_callback_(msg["params"]["data"]);
            }
        }
    }
    else if (msg.contains("id") && msg["id"] == 0) {
        if (msg.contains("result") && msg["result"]["token_type"] == "bearer") {
            std::cout << "Authentication successful" << std::endl;
            subscribe_private_data();
        }
    } 
}

void DeribitClient::send_market_message(const json& msg) {
    {
        std::lock_guard<std::mutex> lock(market_write_mutex_);
        market_message_queue_.push(msg);
    }
    if (!is_market_writing_) {
        write_next_market_message();
    }
}

void DeribitClient::send_trading_message(const json& msg){
    {
        std::lock_guard<std::mutex> lock(trading_write_mutex_);
        trading_message_queue_.push(msg);
    }

    if (!is_trading_writing_) {
        write_next_trading_message();
    }
}

void DeribitClient::write_next_market_message() {
    if (market_message_queue_.empty() || is_market_writing_) {
        return;
    }

    std::lock_guard<std::mutex> lock(market_write_mutex_);
    if (!market_ws_ || !market_connected_) {
        return;
    }

    is_market_writing_ = true;
    auto msg = market_message_queue_.front();
    market_message_queue_.pop();

    market_ws_->async_write(
        net::buffer(msg.dump()),
        [this](const beast::error_code& ec, std::size_t) {
            is_market_writing_ = false;

            if (ec) {
                std::cerr << "Market write error: " << ec.message() << std::endl;
                market_connected_ = false;
                setup_market_connection();
                return;
            }

            // Process next message if any
            if (!market_message_queue_.empty()) {
                write_next_market_message();
            }
        });
}

void DeribitClient::write_next_trading_message() {
    std::unique_lock<std::mutex> lock(trading_write_mutex_);

    if (trading_message_queue_.empty() || is_trading_writing_ || !trading_ws_ || !trading_connected_) {
        return;
    }

    is_trading_writing_ = true;
    auto msg = trading_message_queue_.front();
    trading_message_queue_.pop();

    // Unlock before async_write to prevent deadlock
    lock.unlock();

    trading_ws_->async_write(
        net::buffer(msg.dump()),
        [this](const beast::error_code& ec, std::size_t) {
            std::unique_lock<std::mutex> lock(trading_write_mutex_);
            is_trading_writing_ = false;

            if (ec) {
                std::cerr << "Trading write error: " << ec.message() << std::endl;
                trading_connected_ = false;
                setup_trading_connection();
                return;
            }

            if (!trading_message_queue_.empty()) {
                lock.unlock();  // Unlock before recursive call
                write_next_trading_message();
            }
        });
}

void DeribitClient::handle_error(const std::string& operation, const beast::error_code& ec) {
    std::cerr << operation << " error: " << ec.message() << std::endl;

    if (starts_with(operation, "Market")) {
        {
            std::lock_guard<std::mutex> lock(market_write_mutex_);
            market_connected_ = false;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));  // Prevent excessive reconnection attempts

        {
            std::lock_guard<std::mutex> lock(market_write_mutex_);
            if (!market_connected_) {
                setup_market_connection();
            }
        }
    } else if (starts_with(operation, "Trading")) {
        {
            std::lock_guard<std::mutex> lock(trading_write_mutex_);
            trading_connected_ = false;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));  // Prevent excessive reconnection attempts

        {
            std::lock_guard<std::mutex> lock(trading_write_mutex_);
            if (!trading_connected_) {
                setup_trading_connection();
            }
        }
    }
}
