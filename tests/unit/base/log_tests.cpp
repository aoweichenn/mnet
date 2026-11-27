#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <mnet/core/log.hpp>

namespace {

class RecordingLogger : public mnet::Logger {
public:
    void on_log(const mnet::LogRecord& record) noexcept override { this->records.push_back(record); }

    std::vector<mnet::LogRecord> records;
};

}  // namespace

TEST(Log, DefaultLoggerIsNullAndSilent) {
    EXPECT_EQ(mnet::default_logger(), nullptr);
}

TEST(Log, ScopedDefaultLoggerRestoresPrevious) {
    RecordingLogger first;
    RecordingLogger second;

    {
        const mnet::ScopedDefaultLogger guard{&first};
        EXPECT_EQ(mnet::default_logger(), &first);
        {
            const mnet::ScopedDefaultLogger inner{&second};
            EXPECT_EQ(mnet::default_logger(), &second);
        }
        EXPECT_EQ(mnet::default_logger(), &first);
    }

    EXPECT_EQ(mnet::default_logger(), nullptr);
}

TEST(Log, EveryLevelAndEventHasUniqueName) {
    const mnet::LogLevel levels[] = {
        mnet::LogLevel::kTrace, mnet::LogLevel::kDebug, mnet::LogLevel::kInfo,
        mnet::LogLevel::kWarn,  mnet::LogLevel::kError,
    };
    std::set<std::string> names;
    for (const auto level : levels) {
        const std::string name{mnet::to_string(level)};
        EXPECT_NE(name, "unknown");
        EXPECT_TRUE(names.insert(name).second);
    }

    const mnet::LogEvent events[] = {
        mnet::LogEvent::kStarted,
        mnet::LogEvent::kFinished,
        mnet::LogEvent::kFailure,
    };
    names.clear();
    for (const auto event : events) {
        const std::string name{mnet::to_string(event)};
        EXPECT_NE(name, "unknown");
        EXPECT_TRUE(names.insert(name).second);
    }
}
