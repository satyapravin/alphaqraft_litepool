cmake_minimum_required(VERSION 3.14)
project(LitePool VERSION 1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Boost 1.85 REQUIRED COMPONENTS system filesystem)
find_package(OpenSSL REQUIRED)
include_directories(${PROJECT_SOURCE_DIR} /usr/local/include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

# Enable CTest for test discovery and execution
enable_testing()

add_subdirectory(litepool)
