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
      hedge_symbol_(hedge_symbol.empty() ? symbol_ : std::move(hedge_symbol)),
      instance_id_(generate_instance_id()) {
    std::cout << "Created DeribitClient instance " << instance_id_ << std::endl;
}

DeribitClient::~DeribitClient() {
    std::cout << "Destroying DeribitClient instance " << instance_id_ << std::endl;
    stop();
}

void DeribitClient::start() {
    if (running_.exchange(true)) {
        std::cout << "DeribitClient instance " << instance_id_ << " already running" << std::endl;
        return;
    }

    if (market_ioc_ || trading_ioc_ || market_thread_ || trading_thread_) {
        std::cout << "Cleaning up stale resources before starting instance " << instance_id_ << std::endl;
        stop();
    }

    std::cout << "Starting DeribitClient instance " << instance_id_ << std::endl;
    market_ioc_ = std::make_unique<net::io_context>();
    trading_ioc_ = std::make_unique<net::io_context>();
    ctx_ = std::make_unique<ssl::context>(ssl::context::tlsv12_client);
    market_resolver_ = std::make_unique<tcp::resolver>(*market_ioc_);
    trading_resolver_ = std::make_unique<tcp::resolver>(*trading_ioc_);
    
    ctx_->set_verify_mode(ssl::verify_peer);
    ctx_->set_default_verify_paths();

    market_work_ = std::make_unique<boost::asio::executor_work_guard<net::io_context::executor_type>>(
        market_ioc_->get_executor());
    trading_work_ = std::make_unique<boost::asio::executor_work_guard<net::io_context::executor_type>>(
        trading_ioc_->get_executor());

    setup_connections();
    
    market_thread_ = std::make_unique<std::thread>([this] { 
        std::cout << "Market IO context running for instance " << instance_id_ << std::endl;
        auto work_count = market_ioc_->run();
        std::cout << "Market IO context stopped for instance " << instance_id_ 
                  << " with " << work_count << " operations executed" << std::endl;
    });
    trading_thread_ = std::make_unique<std::thread>([this] { 
        std::cout << "Trading IO context running for instance " << instance_id_ << std::endl;
        auto work_count = trading_ioc_->run();
        std::cout << "Trading IO context stopped for instance " << instance_id_ 
                  << " with " << work_count << " operations executed" << std::endl;
    });
}

void DeribitClient::stop() {
    if (!running_.exchange(false)) {
        std::cout << "DeribitClient instance " << instance_id_ << " already stopped" << std::endl;
        return;
    }

    std::cout << "Stopping DeribitClient instance " << instance_id_ << std::endl;

    boost::system::error_code ec;
    if (market_ws_ && market_connected_) {
        market_ws_->close(websocket::close_code::normal, ec);
        if (ec) std::cerr << "Market close error: " << ec.message() << std::endl;
        market_connected_ = false;
    }
    if (trading_ws_ && trading_connected_) {
        trading_ws_->close(websocket::close_code::normal, ec);
        if (ec) std::cerr << "Trading close error: " << ec.message() << std::endl;
        trading_connected_ = false;
    }

    if (market_timer_) {
        market_timer_->cancel(ec);
        if (ec && ec != boost::asio::error::operation_aborted) {
            std::cerr << "Market timer cancel error: " << ec.message() << std::endl;
        }
    }
    if (trading_timer_) {
        trading_timer_->cancel(ec);
        if (ec && ec != boost::asio::error::operation_aborted) {
            std::cerr << "Trading timer cancel error: " << ec.message() << std::endl;
        }
    }

    if (market_work_) {
        market_work_.reset();
        std::cout << "Market work guard reset" << std::endl;
    }
    if (trading_work_) {
        trading_work_.reset();
        std::cout << "Trading work guard reset" << std::endl;
    }

    if (market_ioc_) {
        market_ioc_->stop();
        std::cout << "Market IO context stopped" << std::endl;
    }
    if (trading_ioc_) {
        trading_ioc_->stop();
        std::cout << "Trading IO context stopped" << std::endl;
    }
    
    if (market_thread_ && market_thread_->joinable()) {
        market_thread_->join();
        std::cout << "Market thread joined" << std::endl;
    }
    if (trading_thread_ && trading_thread_->joinable()) {
        trading_thread_->join();
        std::cout << "Trading thread joined" << std::endl;
    }

    market_ws_.reset();
    trading_ws_.reset();
    market_timer_.reset();
    trading_timer_.reset();
    market_resolver_.reset();
    trading_resolver_.reset();
    market_ioc_.reset();
    trading_ioc_.reset();
    ctx_.reset();
    market_thread_.reset();
    trading_thread_.reset();

    std::cout << "DeribitClient instance " << instance_id_ << " fully stopped" << std::endl;
}

void DeribitClient::setup_connections() {
    setup_market_connection();
    setup_trading_connection();
}

void DeribitClient::setup_market_connection() {
    std::cout << "Setting up market connection for instance " << instance_id_ << std::endl;
    market_ws_ = std::make_unique<websocket_stream>(*market_ioc_, *ctx_);
    market_connected_ = false;
    market_timer_ = std::make_unique<net::steady_timer>(*market_ioc_);
    market_timer_->expires_after(std::chrono::milliseconds(500));
    market_timer_->async_wait([this](const boost::system::error_code& ec) {
        if (!ec && running_) {
            std::cout << "Starting market connection attempt" << std::endl;
            do_market_connect();
        }
    });
}

void DeribitClient::setup_trading_connection() {
    std::cout << "Setting up trading connection for instance " << instance_id_ << std::endl;
    trading_ws_ = std::make_unique<websocket_stream>(*trading_ioc_, *ctx_);
    trading_connected_ = false;
    trading_timer_ = std::make_unique<net::steady_timer>(*trading_ioc_);
    trading_timer_->expires_after(std::chrono::milliseconds(500));
    trading_timer_->async_wait([this](const boost::system::error_code& ec) {
        if (!ec && running_) {
            std::cout << "Starting trading connection attempt" << std::endl;
            do_trading_connect();
        }
    });
}

void DeribitClient::do_market_connect() {
    std::cout << "Attempting market connection" << std::endl;
    constexpr auto host = "www.deribit.com";
    constexpr auto port = "443";

    std::cout << "Starting market resolve for " << host << ":" << port << std::endl;
    market_resolver_->async_resolve(host, port, 
        [this](const boost::system::error_code& ec, tcp::resolver::results_type results) {
            if (ec) {
                std::cout << "Market resolve failed: " << ec.message() << std::endl;
                return handle_error("Market Resolve", ec);
            }
            std::cout << "Market resolve succeeded" << std::endl;
            
            auto& lowest_layer = beast::get_lowest_layer(*market_ws_);
            std::cout << "Starting market async_connect" << std::endl;
            boost::asio::async_connect(
                lowest_layer,
                results,
                [this](const boost::system::error_code& ec, const tcp::endpoint&) {
                    if (ec) {
                        std::cout << "Market connect failed: " << ec.message() << std::endl;
                        return handle_error("Market Connect", ec);
                    }
                    std::cout << "Market connect succeeded" << std::endl;
                    
                    market_ws_->next_layer().async_handshake(
                        ssl::stream_base::client,
                        [this](const boost::system::error_code& ec) {
                            if (ec) {
                                std::cout << "Market SSL handshake failed: " << ec.message() << std::endl;
                                return handle_error("Market SSL", ec);
                            }
                            std::cout << "Market SSL handshake succeeded" << std::endl;
                            
                            market_ws_->async_handshake(
                                host, "/ws/api/v2",
                                [this](const boost::system::error_code& ec) {
                                    if (ec) {
                                        std::cout << "Market WS handshake failed: " << ec.message() << std::endl;
                                        return handle_error("Market WS", ec);
                                    }
                                    std::cout << "Market WS handshake succeeded" << std::endl;
                                    
                                    market_connected_ = true;
                                    subscribe_market_data();
                                    do_market_read();
                                });
                        });
                });
        });
}

void DeribitClient::do_trading_connect() {
    std::cout << "Attempting trading connection" << std::endl;
    constexpr auto host = "www.deribit.com";
    constexpr auto port = "443";

    std::cout << "Starting trading resolve for " << host << ":" << port << std::endl;
    trading_resolver_->async_resolve(host, port,
        [this](const boost::system::error_code& ec, tcp::resolver::results_type results) {
            if (ec) {
                std::cout << "Trading resolve failed: " << ec.message() << std::endl;
                return handle_error("Trading Resolve", ec);
            }
            std::cout << "Trading resolve succeeded" << std::endl;
            
            auto& lowest_layer = beast::get_lowest_layer(*trading_ws_);
            std::cout << "Starting trading async_connect" << std::endl;
            boost::asio::async_connect(
                lowest_layer,
                results,
                [this](const boost::system::error_code& ec, const tcp::endpoint&) {
                    if (ec) {
                        std::cout << "Trading connect failed: " << ec.message() << std::endl;
                        return handle_error("Trading Connect", ec);
                    }
                    std::cout << "Trading connect succeeded" << std::endl;
                    
                    trading_ws_->next_layer().async_handshake(
                        ssl::stream_base::client,
                        [this](const boost::system::error_code& ec) {
                            if (ec) {
                                std::cout << "Trading SSL handshake failed: " << ec.message() << std::endl;
                                return handle_error("Trading SSL", ec);
                            }
                            std::cout << "Trading SSL handshake succeeded" << std::endl;
                            
                            trading_ws_->async_handshake(
                                host, "/ws/api/v2",
                                [this](const boost::system::error_code& ec) {
                                    if (ec) {
                                        std::cout << "Trading WS handshake failed: " << ec.message() << std::endl;
                                        return handle_error("Trading WS", ec);
                                    }
                                    std::cout << "Trading WS handshake succeeded" << std::endl;
                                    
                                    trading_connected_ = true;
                                    authenticate();
                                    do_trading_read();
                                });
                        });
                });
        });
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
    try {
        if (!msg.contains("method") || msg["method"] != "subscription") {
            return;
        }
        if (!msg.contains("params") || !msg["params"].contains("channel") || !msg["params"].contains("data")) {
            std::cerr << "Invalid market message format for instance " << instance_id_ << std::endl;
            return;
        }

        const auto& channel = msg["params"]["channel"].get<std::string>();
        if (channel.find("book.") == 0 && OrderBook_cb_) {
            OrderBook_cb_(msg["params"]["data"]);
        }
    } catch (const std::exception& e) {
        std::cerr << "Market message handling error for instance " << instance_id_ << ": " << e.what() << std::endl;
    }
}

void DeribitClient::handle_trading_message(const json& msg) {
    try {
        if (msg.contains("method") && msg["method"] == "subscription") {
            if (!msg.contains("params") || !msg["params"].contains("channel") || !msg["params"].contains("data")) {
                std::cerr << "Invalid trading message format for instance " << instance_id_ << std::endl;
                return;
            }
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
                 msg.contains("result") && msg["result"].contains("token_type") && 
                 msg["result"]["token_type"] == "bearer") {
            std::cout << "Authentication successful for instance " << instance_id_ << ", subscribing to private data" << std::endl;
            subscribe_private_data();
        }
    } catch (const std::exception& e) {
        std::cerr << "Trading message handling error for instance " << instance_id_ << ": " << e.what() << std::endl;
    }
}

void DeribitClient::send_market_message(json&& msg) {
    if (!market_out_queue_.push(std::move(msg))) {
        std::cerr << "Market output queue full for instance " << instance_id_ << std::endl;
        return;
    }
    net::post(*market_ioc_, [this] { write_next_market_message(); });
}

void DeribitClient::send_trading_message(json&& msg) {
    if (!trading_out_queue_.push(std::move(msg))) {
        std::cerr << "Trading output queue full for instance " << instance_id_ << std::endl;
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
    std::cerr << context << " error for instance " << instance_id_ << ": " << ec.message() << std::endl;
    
    if (ec == boost::asio::error::operation_aborted) {
        std::cerr << context << ": operation canceled, skipping retry (running=" << running_ << ")" << std::endl;
        return;
    }
    
    if (context.find("Market") != std::string::npos) {
        market_connected_ = false;
        if (running_) {
            std::cout << "Scheduling market reconnect in 5 seconds for instance " << instance_id_ << std::endl;
            market_timer_ = std::make_unique<net::steady_timer>(*market_ioc_);
            market_timer_->expires_after(std::chrono::seconds(5));
            market_timer_->async_wait([this](const boost::system::error_code& ec) {
                if (!ec && running_) {
                    std::cout << "Attempting market reconnect for instance " << instance_id_ << std::endl;
                    setup_market_connection();
                }
            });
        }
    } 
    else if (context.find("Trading") != std::string::npos) {
        trading_connected_ = false;
        if (running_) {
            std::cout << "Scheduling trading reconnect in 5 seconds for instance " << instance_id_ << std::endl;
            trading_timer_ = std::make_unique<net::steady_timer>(*trading_ioc_);
            trading_timer_->expires_after(std::chrono::seconds(5));
            trading_timer_->async_wait([this](const boost::system::error_code& ec) {
                if (!ec && running_) {
                    std::cout << "Attempting trading reconnect for instance " << instance_id_ << std::endl;
                    setup_trading_connection();
                }
            });
        }
    }
}

} // namespace RLTrader
