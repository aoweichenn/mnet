// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

// Internal diagnostics sink. Library code must not print directly; route
// through this narrow helper so the transport can be replaced in one place.

#ifndef MNET_DETAIL_TXN_LOG_HPP
#define MNET_DETAIL_TXN_LOG_HPP

#include <cstdio>

#include <string_view>

namespace mnet::detail {

inline void write_diagnostic(std::string_view line) {
    if (line.empty()) {
        return;
    }
    static_cast<void>(std::fwrite(line.data(), 1, line.size(), stderr));
}

}  // namespace mnet::detail

#endif  // MNET_DETAIL_TXN_LOG_HPP
