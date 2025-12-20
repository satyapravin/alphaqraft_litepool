cmake_minimum_required(VERSION 3.14)
project(LitePool VERSION 1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Use Python from environment if set (allows users to specify which Python to use)
# Otherwise, prefer python3.11, then fall back to python3
if(DEFINED ENV{PYTHON_EXECUTABLE})
    set(Python3_EXECUTABLE $ENV{PYTHON_EXECUTABLE} CACHE FILEPATH "Python 3 executable" FORCE)
    message(STATUS "Using Python from PYTHON_EXECUTABLE: ${Python3_EXECUTABLE}")
elseif(DEFINED ENV{PYTHON})
    set(Python3_EXECUTABLE $ENV{PYTHON} CACHE FILEPATH "Python 3 executable" FORCE)
    message(STATUS "Using Python from PYTHON: ${Python3_EXECUTABLE}")
else()
    # Try to find python3.11 first (prefer newer versions), then fall back to python3
    find_program(PYTHON3_11_EXECUTABLE NAMES python3.11 python3.12 python3.13)
    if(PYTHON3_11_EXECUTABLE)
        set(Python3_EXECUTABLE ${PYTHON3_11_EXECUTABLE} CACHE FILEPATH "Python 3 executable" FORCE)
        message(STATUS "Found Python 3.11+: ${Python3_EXECUTABLE}")
    endif()
endif()

# Set Python3_ROOT_DIR based on executable to help find_package locate the right Python
if(Python3_EXECUTABLE)
    get_filename_component(PYTHON3_POTENTIAL_ROOT ${Python3_EXECUTABLE} DIRECTORY)
    get_filename_component(PYTHON3_POTENTIAL_ROOT ${PYTHON3_POTENTIAL_ROOT} DIRECTORY)
    if(EXISTS "${PYTHON3_POTENTIAL_ROOT}/include")
        set(Python3_ROOT_DIR ${PYTHON3_POTENTIAL_ROOT} CACHE PATH "Python 3 root directory" FORCE)
    endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Log which Python version was found (for debugging)
if(Python3_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} --version
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    message(STATUS "Python version: ${PYTHON_VERSION}")
    message(STATUS "Python executable: ${Python3_EXECUTABLE}")
endif()
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
