#include <set>
#include <string>

#include <gtest/gtest.h>
#include <mnet/error.hpp>

TEST(Error, EveryValueHasUniqueName) {
    const mnet::Error values[] = {
        mnet::Error::kNone, mnet::Error::kInvalidArgument, mnet::Error::kMalformed, mnet::Error::kUnsupported,
        mnet::Error::kIo,   mnet::Error::kOutOfMemory,
    };

    std::set<std::string> seen;
    for (const auto value : values) {
        const std::string name{mnet::to_string(value)};
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name, "unknown");
        EXPECT_TRUE(seen.insert(name).second) << "两个错误码用了同一个名字: " << name;
    }
}
