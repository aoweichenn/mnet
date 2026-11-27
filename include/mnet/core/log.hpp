// mnet 公共契约头：注入式诊断。
//
// 库不持有日志运行时：只定义"发生了什么"（LogRecord），宿主实现 Logger
// 决定落盘方式。未接线时零输出、零依赖。
//
// 生效时机：对象在构造时捕获出口；set_default_logger() 只影响之后创建的对象。

#ifndef MNET_LOG_HPP
#define MNET_LOG_HPP

#include <cstddef>
#include <cstdint>

#include <source_location>
#include <string_view>

#include <mnet/core/error.hpp>

namespace mnet {

enum class LogLevel : std::uint8_t {
    kTrace = 0,
    kDebug,
    kInfo,
    kWarn,
    kError,
};

enum class LogEvent : std::uint8_t {
    kStarted = 0,
    kFinished = 1,
    kFailure = 2,
};

struct LogRecord {
    LogEvent event = LogEvent::kStarted;
    LogLevel level = LogLevel::kInfo;
    std::source_location location{};

    Error code = Error::kNone;
    std::uint64_t offset = 0;
    std::uint64_t declared = 0;
    std::string_view detail;
    std::size_t item_count = 0;
};

class Logger {
public:
    Logger() = default;
    virtual ~Logger() = default;

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept { return "mnet"; }

    [[nodiscard]] virtual bool should_log(LogLevel level) const noexcept {
        (void)level;
        return true;
    }

    virtual void on_log(const LogRecord& record) noexcept = 0;
};

void set_default_logger(Logger* logger) noexcept;

[[nodiscard]] Logger* default_logger() noexcept;

// RAII：构造保存旧值并设置，析构恢复。只支持单线程内嵌套。
class ScopedDefaultLogger {
public:
    explicit ScopedDefaultLogger(Logger* logger) noexcept;
    ~ScopedDefaultLogger();

    ScopedDefaultLogger(const ScopedDefaultLogger&) = delete;
    ScopedDefaultLogger& operator=(const ScopedDefaultLogger&) = delete;
    ScopedDefaultLogger(ScopedDefaultLogger&&) = delete;
    ScopedDefaultLogger& operator=(ScopedDefaultLogger&&) = delete;

private:
    Logger* previous_ = nullptr;
};

[[nodiscard]] std::string_view to_string(LogLevel level) noexcept;
[[nodiscard]] std::string_view to_string(LogEvent event) noexcept;

}  // namespace mnet

#endif  // MNET_LOG_HPP
