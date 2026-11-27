// 默认 Logger 只是一个 atomic 指针：没有 init/shutdown，没有线程与文件。

#include <mnet/core/log.hpp>

#include <atomic>

namespace mnet {

namespace {

std::atomic<Logger*> g_default_logger{nullptr};

static_assert(std::atomic<Logger*>::is_always_lock_free, "default logger pointer must be lock-free");

}  // namespace

void set_default_logger(Logger* logger) noexcept {
    g_default_logger.store(logger, std::memory_order_release);
}

Logger* default_logger() noexcept {
    return g_default_logger.load(std::memory_order_acquire);
}

ScopedDefaultLogger::ScopedDefaultLogger(Logger* logger) noexcept
        : previous_(g_default_logger.exchange(logger, std::memory_order_acq_rel)) {}

ScopedDefaultLogger::~ScopedDefaultLogger() {
    g_default_logger.store(this->previous_, std::memory_order_release);
}

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::kTrace:
            return "trace";
        case LogLevel::kDebug:
            return "debug";
        case LogLevel::kInfo:
            return "info";
        case LogLevel::kWarn:
            return "warn";
        case LogLevel::kError:
            return "error";
    }
    return "unknown";
}

std::string_view to_string(LogEvent event) noexcept {
    switch (event) {
        case LogEvent::kStarted:
            return "started";
        case LogEvent::kFinished:
            return "finished";
        case LogEvent::kFailure:
            return "failure";
    }
    return "unknown";
}

}  // namespace mnet
