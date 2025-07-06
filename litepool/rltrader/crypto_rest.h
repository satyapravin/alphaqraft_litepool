// crypto_rest.h
#pragma once
#include <string>

namespace RLTrader {
    class CryptoREST {
    public:
        CryptoREST(const std::string& key, const std::string& secret)
            : api_key_(key), api_secret_(secret) {}
        // Fills amount & avgPrice for given instrument. Returns true on success.
        bool fetch_position(const std::string& symbol, double& amount, double& avgPrice);

    private:
        std::string api_key_;
        std::string api_secret_;
    };
}
