#include <cstdint>

#include <utility>

#include <gtest/gtest.h>
#include <mnet/core/result.hpp>

TEST(Status, DefaultIsOk) {
    const mnet::Status status{};
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(static_cast<bool>(status));
}

TEST(Status, CarriesFailure) {
    const mnet::Status status{mnet::Failure{mnet::Error::kIo, 7, "boom"}};
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.failure().code, mnet::Error::kIo);
    EXPECT_EQ(status.failure().offset, 7U);
}

TEST(Result, MoveTransfersValue) {
    mnet::Result<std::int32_t> first{42};
    ASSERT_TRUE(first.ok());

    mnet::Result<std::int32_t> second{std::move(first)};
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(second.value(), 42);
}

TEST(Result, FailureCarriesOffsetAndDeclared) {
    const mnet::Failure failure{mnet::Error::kMalformed, 3, 99, "bad"};
    EXPECT_EQ(failure.offset, 3U);
    EXPECT_EQ(failure.declared, 99U);
}
