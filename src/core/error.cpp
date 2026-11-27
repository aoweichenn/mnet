// 错误码的稳定字符串名。日志与测试都依赖它，不允许随版本漂移。

#include <mnet/core/error.hpp>

namespace mnet {

std::string_view to_string(Error code) noexcept {
    switch (code) {
        case Error::kNone:
            return "none";
        case Error::kInvalidArgument:
            return "invalid_argument";
        case Error::kMalformed:
            return "malformed";
        case Error::kUnsupported:
            return "unsupported";
        case Error::kIo:
            return "io";
        case Error::kOutOfMemory:
            return "out_of_memory";
    }
    return "unknown";
}

}  // namespace mnet
