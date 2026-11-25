// mnet 公共契约头：错误码与错误载荷。

#ifndef MNET_ERROR_HPP
#define MNET_ERROR_HPP

#include <cstdint>

#include <string_view>

namespace mnet {

// 闭集：每个失败路径映射到唯一错误码；测试枚举全部值并断言名字唯一。
enum class Error : std::uint8_t {
    kNone = 0,
    kInvalidArgument,
    kMalformed,
    kUnsupported,
    kIo,
    kOutOfMemory,
};

[[nodiscard]] std::string_view to_string(Error code) noexcept;

// 失败载荷：where（可定位偏移）与 what（元数据里的坏值）分开。
struct Failure {
    Error code = Error::kNone;
    std::uint64_t offset = 0;
    std::uint64_t declared = 0;
    std::string_view detail;  // 静态字面量

    Failure() noexcept = default;
    Failure(Error error, std::uint64_t at, std::string_view text) noexcept : code(error), offset(at), detail(text) {}
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Failure(Error error, std::uint64_t at, std::uint64_t value, std::string_view text) noexcept
            : code(error), offset(at), declared(value), detail(text) {}
};

}  // namespace mnet

#endif  // MNET_ERROR_HPP
