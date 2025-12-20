// Copyright 2024 Alphaqraft
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cmath>
#include <vector>

#define NORMALIZE(struct_name, to_struct, mean_name, ssr_name, property_name, alpha) \
    do { \
        mean_name->property_name *= (1.0 - alpha); \
        mean_name->property_name += alpha * (struct_name.property_name - mean_name->property_name); \
        auto sqdelta = struct_name.property_name - mean_name->property_name; \
        ssr_name->property_name *= (1.0 - alpha); \
        ssr_name->property_name += alpha * sqdelta * sqdelta; \
        to_struct.property_name = sqdelta / std::pow(ssr_name->property_name + 1e-12, 0.5); \
    } while(0)

template<typename T>
void insert_signals(std::vector<double>& signals, T& repo) {
    size_t num_doubles = sizeof(T) / sizeof(double);
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Waddress-of-packed-member"
    auto* start = reinterpret_cast<double*>(&repo);
    #pragma GCC diagnostic pop
    double* end = start + num_doubles;
    signals.insert(signals.end(), start, end);
}