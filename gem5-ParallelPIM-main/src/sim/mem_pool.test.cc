/*
 * Unit test of MemPool::contains and MemPools::poolOf, which
 * Process::deallocateMem uses to leave frames an emulated driver maps
 * (PimMemDriver) out of the SE page pools.
 */

#include <gtest/gtest.h>

#include "base/addr_range.hh"
#include "sim/mem_pool.hh"

using namespace gem5;

namespace
{

constexpr Addr PageShift = 12;
constexpr Addr PageBytes = Addr(1) << PageShift;

} // anonymous namespace

TEST(MemPoolTest, Contains)
{
    MemPool pool(PageShift, 0x10000, 0x20000);
    EXPECT_FALSE(pool.contains(0xffff));
    EXPECT_TRUE(pool.contains(0x10000));
    EXPECT_TRUE(pool.contains(0x10fff));
    EXPECT_TRUE(pool.contains(0x1ffff));
    EXPECT_FALSE(pool.contains(0x20000));
}

TEST(MemPoolsTest, PoolOf)
{
    MemPools pools(PageShift);
    pools.populate({AddrRange(0, 0x100000),
                    AddrRange(0x200000, 0x300000)});
    EXPECT_EQ(pools.poolOf(0), 0);
    EXPECT_EQ(pools.poolOf(0x100000 - PageBytes), 0);
    EXPECT_EQ(pools.poolOf(0x100000), -1);
    EXPECT_EQ(pools.poolOf(0x200000), 1);
    EXPECT_EQ(pools.poolOf(0x300000 - PageBytes), 1);
    EXPECT_EQ(pools.poolOf(0x300000), -1);
    // Memory outside every pool, such as a window an emulated driver
    // maps (PimMemDriver), belongs to none.
    EXPECT_EQ(pools.poolOf(0x400000000), -1);
}

TEST(MemPoolsTest, PoolOfAllocatedPages)
{
    MemPools pools(PageShift);
    pools.populate({AddrRange(0, 0x100000),
                    AddrRange(0x200000, 0x300000)});
    const Addr a = pools.allocPhysPages(1, 0);
    const Addr b = pools.allocPhysPages(1, 1);
    EXPECT_EQ(pools.poolOf(a), 0);
    EXPECT_EQ(pools.poolOf(b), 1);
    pools.deallocPhysPages(b, 1, pools.poolOf(b));
    EXPECT_EQ(pools.freeMemSize(1), pools.memSize(1));
    EXPECT_EQ(pools.freeMemSize(0), pools.memSize(0) - PageBytes);
}
