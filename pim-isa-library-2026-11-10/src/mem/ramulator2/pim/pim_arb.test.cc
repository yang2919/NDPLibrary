/*
 * Tests of the PIM arbitration policies (docs/pim/04-memctrl-plan.md
 * sections 7.2 and 12.1; plan docs/pim/04r-ramulator-pim-plan.md section
 * 4.7), the pure code that the Ramulator 2.1 sequencer
 * (ext/ramulator2/src/ramulator/pim/pim_arb.h) asks every cycle.
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "ramulator2/src/ramulator/pim/pim_arb.h"

using namespace Ramulator;

namespace
{

PimArbView
view(bool pending, bool ready, bool started, bool pim_hit, bool queued,
     bool host_ready, bool host_hit, bool older)
{
    PimArbView v;
    v.pim_pending = pending;
    v.pim_ready = ready;
    v.pim_started = started;
    v.pim_row_hit = pim_hit;
    v.host_queued = queued;
    v.host_ready = host_ready;
    v.host_row_hit = host_hit;
    v.host_older = older;
    return v;
}

} // anonymous namespace

TEST(PimArb, FrfcfsTable)
{
    const auto p = make_pim_arb_policy("frfcfs");
    EXPECT_EQ(p->gate(), PimGate::WhileStarted);
    // A started request is never interrupted (P4 Q3 (a)), whatever the
    // host side offers.
    EXPECT_EQ(p->select(view(1, 0, 1, 0, 1, 1, 1, 1)), PimSide::Pim);
    // Not ready: the host.
    EXPECT_EQ(p->select(view(1, 0, 0, 1, 1, 0, 0, 0)), PimSide::Host);
    // Ready and the host is not: PIM.
    EXPECT_EQ(p->select(view(1, 1, 0, 0, 1, 0, 0, 1)), PimSide::Pim);
    // Both ready: the row hit wins.
    EXPECT_EQ(p->select(view(1, 1, 0, 1, 1, 1, 0, 1)), PimSide::Pim);
    EXPECT_EQ(p->select(view(1, 1, 0, 0, 1, 1, 1, 0)), PimSide::Host);
    // Both ready, both hits or both misses: the older request.
    EXPECT_EQ(p->select(view(1, 1, 0, 1, 1, 1, 1, 1)), PimSide::Host);
    EXPECT_EQ(p->select(view(1, 1, 0, 1, 1, 1, 1, 0)), PimSide::Pim);
    EXPECT_EQ(p->select(view(1, 1, 0, 0, 1, 1, 0, 1)), PimSide::Host);
    EXPECT_EQ(p->select(view(1, 1, 0, 0, 1, 1, 0, 0)), PimSide::Pim);
}

TEST(PimArb, ChopimTable)
{
    const auto p = make_pim_arb_policy("chopim");
    EXPECT_EQ(p->gate(), PimGate::Never);
    // The host first, before and inside a request.
    EXPECT_EQ(p->select(view(1, 1, 0, 1, 1, 0, 0, 0)), PimSide::Host);
    EXPECT_EQ(p->select(view(1, 1, 1, 1, 1, 0, 0, 0)), PimSide::Host);
    EXPECT_EQ(p->select(view(1, 1, 1, 0, 0, 0, 0, 0)), PimSide::Pim);
    EXPECT_EQ(p->select(view(1, 0, 0, 0, 0, 0, 0, 0)), PimSide::Pim);
}

TEST(PimArb, AsyncDimmTable)
{
    const auto p = make_pim_arb_policy("asyncdimm");
    EXPECT_EQ(p->gate(), PimGate::WhileSelected);
    PimArbView v = view(1, 1, 0, 0, 1, 1, 0, 0);
    // On the host side with host work: stay, until a PRE.
    v.last_side = PimSide::Host;
    v.last_was_pre = false;
    EXPECT_EQ(p->select(v), PimSide::Host);
    v.last_was_pre = true;
    EXPECT_EQ(p->select(v), PimSide::Pim);
    // On the PIM side: stay until a PRE, if the host has work.
    v.last_side = PimSide::Pim;
    v.last_was_pre = false;
    EXPECT_EQ(p->select(v), PimSide::Pim);
    v.last_was_pre = true;
    EXPECT_EQ(p->select(v), PimSide::Host);
    // A PRE with nothing on the other side does not switch.
    v.host_queued = false;
    EXPECT_EQ(p->select(v), PimSide::Pim);
    // The current side has nothing: switch.
    v.pim_pending = false;
    v.host_queued = true;
    v.last_was_pre = false;
    EXPECT_EQ(p->select(v), PimSide::Host);
    v.last_side = PimSide::Host;
    v.host_queued = false;
    v.pim_pending = true;
    EXPECT_EQ(p->select(v), PimSide::Pim);
}

TEST(PimArb, CosmAndUnknown)
{
    EXPECT_THROW(make_pim_arb_policy("cosm"), std::runtime_error);
    EXPECT_THROW(make_pim_arb_policy("fifo"), std::runtime_error);
}
