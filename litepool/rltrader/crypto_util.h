// -----------------------------------------------------------------------------
// utils.h  (add to your project)
// -----------------------------------------------------------------------------
#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <algorithm>

namespace RLTrader
{
    using json = nlohmann::json;

    // Recursively flattens an object/array according to Crypto.com spec
    inline std::string flatten_params(const json& j)
    {
        if (j.is_null())      return "";                          // null -> ""
        if (j.is_boolean())   return j.get<bool>() ? "true" : "false";
        if (j.is_string())    return j.get<std::string>();
        if (j.is_number())    return j.dump();                    // keeps full precision

        if (j.is_array()) {                                       // concatenate elements
            std::string out;
            for (const auto& v : j) out += flatten_params(v);
            return out;
        }
        if (j.is_object()) {                                      // sort keys first
            std::vector<std::string> keys;
            keys.reserve(j.size());
            for (auto it = j.begin(); it != j.end(); ++it)
                keys.push_back(it.key());
            std::sort(keys.begin(), keys.end());

            std::string out;
            for (const auto& k : keys) {
                out += k;
                out += flatten_params(j.at(k));
            }
            return out;
        }
        return "";
    }

    // Builds the full pre-hash payload
    inline std::string build_payload(const std::string& method,
                                     const int id,
                                     const std::string& api_key,
                                     const json&        params,
                                     long               nonce)
    {
        return method + id + api_key + flatten_params(params) + std::to_string(nonce);
    }
} // namespace RLTrader
