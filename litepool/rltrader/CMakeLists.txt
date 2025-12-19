include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)
find_package(GTest REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)
set(OPENSSL_ROOT_DIR "/usr/openssl")
set(OPENSSL_LIBRARIES "/lib64/libssl.so;/lib64/libcrypto.so")
find_package(OpenSSL REQUIRED)

add_executable(testcases testcases.cc
        csv_reader.h csv_reader.cc
        sim_exchange.h sim_exchange.cc
        inverse_instrument.h inverse_instrument.cc
        normal_instrument.h normal_instrument.cc
        order.h position.h position.cc
        strategy.cc doctest.h strategy.h
        orderbook.h orderbook_buffer.h
        market_signal_builder.h market_signal_builder.cc
        trade_reader.h trade_reader.cc
        trade_signal_builder.h trade_signal_builder.cc
        env_adaptor.h env_adaptor.cc
        amm_simulator.h amm_simulator.cc
        circ_buffer.h circ_table.h
        fixed_vector.h)

set(GFLAG_LIBRARY_NAME /usr/local/lib/libgflags.a)
pybind11_add_module(rltrader_litepool rltrader_litepool.h rltrader_litepool.cc
                                      base_instrument.h rl_macros.h
                                      csv_reader.h csv_reader.cc
                                      circ_buffer.h circ_table.h
                                      base_exchange.h
                                      sim_exchange.h sim_exchange.cc
                                      crypto_exchange.h crypto_exchange.cc
                                      crypto_client.h crypto_client.cc
                                      crypto_rest.h crypto_rest.cc crypto_util.h
                                      deribit_exchange.h deribit_exchange.cc
                                      deribit_client.h deribit_client.cc
                                      deribit_rest.h deribit_rest.cc
                                      inverse_instrument.h inverse_instrument.cc
                                      normal_instrument.h normal_instrument.cc
                                      order.h position.h position.cc
                                      strategy.cc doctest.h strategy.h
                                      orderbook.h orderbook_buffer.h
                                      market_signal_builder.h market_signal_builder.cc
                                      trade_reader.h trade_reader.cc
                                      trade_signal_builder.h trade_signal_builder.cc
                                      env_adaptor.h env_adaptor.cc
                                      amm_simulator.h amm_simulator.cc
                                      norm_macro.h rl_macros.h
)

set_target_properties(rltrader_litepool PROPERTIES PREFIX "" SUFFIX ".so")
include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(temptest temp.cc)

add_executable(rltradertest rltrader_litepool_test.cc
                                      base_instrument.h rl_macros.h
                                      csv_reader.h csv_reader.cc
                                      base_exchange.h
                                      circ_table.h circ_buffer.h
                                      sim_exchange.h sim_exchange.cc
                                      crypto_exchange.h crypto_exchange.cc
                                      crypto_client.h crypto_client.cc
                                      crypto_rest.h crypto_rest.cc crypto_util.h
                                      deribit_exchange.h deribit_exchange.cc
                                      deribit_client.h deribit_client.cc
                                      deribit_rest.h deribit_rest.cc
                                      inverse_instrument.h inverse_instrument.cc
                                      normal_instrument.h normal_instrument.cc
                                      order.h position.h position.cc
                                      strategy.cc doctest.h strategy.h
                                      orderbook.h orderbook_buffer.h
                                      market_signal_builder.h market_signal_builder.cc
                                      trade_reader.h trade_reader.cc
                                      trade_signal_builder.h trade_signal_builder.cc
                                      env_adaptor.h env_adaptor.cc
                                      amm_simulator.h amm_simulator.cc
                                      norm_macro.h rl_macros.h
                                      rl_macros.h
                                      norm_macro.h
                                      orderbook_buffer.h
                                      fixed_vector.h
                                      fixed_vector.h
)

target_link_libraries(temptest PUBLIC OpenSSL::SSL OpenSSL::Crypto Boost::system Threads::Threads
                                        PRIVATE glog -lstdc++fs nlohmann_json::nlohmann_json)
target_link_libraries(rltrader_litepool PUBLIC OpenSSL::SSL OpenSSL::Crypto Boost::system Threads::Threads
                                        PRIVATE glog -lstdc++fs nlohmann_json::nlohmann_json)
target_link_libraries(rltradertest PUBLIC OpenSSL::SSL OpenSSL::Crypto Boost::system Threads::Threads
                                   PRIVATE GTest::GTest GTest::Main gflags gmock glog -lstdc++fs nlohmann_json::nlohmann_json)
target_link_libraries(testcases PUBLIC OpenSSL::SSL OpenSSL::Crypto Boost::system Threads::Threads
                                   PRIVATE glog -lstdc++fs nlohmann_json::nlohmann_json)

# Register doctest testcases executable with CTest
add_test(NAME doctest_testcases COMMAND testcases)

# Configure test properties
# Set working directory to project root so test_data/ paths resolve correctly
set_tests_properties(doctest_testcases PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Add custom target to run tests (can be invoked with: make test)
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS testcases rltradertest
    COMMENT "Running all tests..."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

include(GoogleTest)
gtest_discover_tests(rltradertest)
