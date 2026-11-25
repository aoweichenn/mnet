// Copyright 2026 mcnn authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <mmemory/allocator.hpp>

namespace {

TEST(BudgetAllocator, AccountsCurrentAndPeak) {
    mmemory::PoolAllocator pool;
    mmemory::BudgetAllocator budget(pool, 0);

    void* a = budget.allocate(1000);
    void* b = budget.allocate(2000);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(budget.current_bytes(), 3000U);
    EXPECT_EQ(budget.peak_bytes(), 3000U);

    budget.deallocate(b);
    EXPECT_EQ(budget.current_bytes(), 1000U);
    EXPECT_EQ(budget.peak_bytes(), 3000U);

    budget.deallocate(a);
    EXPECT_EQ(budget.current_bytes(), 0U);
}

TEST(BudgetAllocator, RefusesRequestOverBudget) {
    mmemory::PoolAllocator pool;
    mmemory::BudgetAllocator budget(pool, 1500);

    void* a = budget.allocate(1000);
    ASSERT_NE(a, nullptr);
    EXPECT_FALSE(budget.exhausted());

    void* b = budget.allocate(1000);
    EXPECT_EQ(b, nullptr);
    EXPECT_TRUE(budget.exhausted());
    EXPECT_EQ(budget.current_bytes(), 1000U);

    // Releasing space makes room again; the exhausted flag is sticky so the
    // caller can report that the graph already failed once.
    budget.deallocate(a);
    void* c = budget.allocate(1400);
    EXPECT_NE(c, nullptr);
    EXPECT_TRUE(budget.exhausted());
    budget.deallocate(c);
}

TEST(BudgetAllocator, PreservesAlignment) {
    mmemory::PoolAllocator pool;
    mmemory::BudgetAllocator budget(pool, 0);

    for (std::size_t size : {1U, 7U, 16U, 333U, 4096U}) {
        void* ptr = budget.allocate(size);
        ASSERT_NE(ptr, nullptr);
        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        EXPECT_EQ(address % mmemory::kMallocAlignment, 0U);
        budget.deallocate(ptr);
    }
    EXPECT_EQ(budget.current_bytes(), 0U);
}

TEST(BudgetAllocator, SetBudgetTakesEffect) {
    mmemory::PoolAllocator pool;
    mmemory::BudgetAllocator budget(pool, 100);
    void* a = budget.allocate(80);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(budget.allocate(80), nullptr);

    budget.set_budget(200);
    void* b = budget.allocate(80);
    EXPECT_NE(b, nullptr);
    budget.deallocate(b);
    budget.deallocate(a);
    EXPECT_EQ(budget.current_bytes(), 0U);
}

}  // namespace
