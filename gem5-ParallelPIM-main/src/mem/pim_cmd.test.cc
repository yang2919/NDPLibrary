/*
 * Unit test of the processing-in-memory (PIM) request interface and
 * memory commands.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "base/gtest/cur_tick_fake.hh"
#include "base/gtest/logging.hh"
#include "mem/packet.hh"
#include "mem/pim_op.hh"
#include "mem/request.hh"

using namespace gem5;

// Requests read curTick(); a fake tick of 0.
GTestTickHandler tickHandler;

namespace
{

constexpr Addr Base = 0x10000;
constexpr unsigned Size = 64;

PimOp
op(unsigned i)
{
    return static_cast<PimOp>(i);
}

Request::Flags
pimFlags(PimOp o)
{
    return Request::PIM | Request::UNCACHEABLE | Request::STRICT_ORDER |
        Request::pimOpFlags(o);
}

RequestPtr
pimRequest(PimOp o, Request::Flags extra = 0)
{
    return std::make_shared<Request>(Base, Size, pimFlags(o) | extra, 0);
}

std::string
pimName(PimOp o)
{
    return std::string("Pim") + pimOpName(o);
}

} // anonymous namespace

/** Every operation survives a round trip through the request flags. */
TEST(PimRequest, OperationRoundTrip)
{
    for (unsigned i = 1; i <= NumPimOps; i++) {
        auto req = pimRequest(op(i));
        EXPECT_TRUE(req->isPim());
        EXPECT_EQ(req->getPimOp(), op(i));
        EXPECT_TRUE(req->isUncacheable());
        EXPECT_TRUE(req->isStrictlyOrdered());
    }
    auto plain = std::make_shared<Request>(
        Base, Size, Request::UNCACHEABLE | Request::STRICT_ORDER, 0);
    EXPECT_FALSE(plain->isPim());
    EXPECT_EQ(plain->getPimOp(), PimOp::None);
}

/** The operation table: names, load-like operations. */
TEST(PimRequest, Names)
{
    EXPECT_STREQ(pimOpName(PimOp::None), "None");
    EXPECT_STREQ(pimOpName(PimOp::SbEwadd), "SbEwadd");
    EXPECT_STREQ(pimOpName(PimOp::AbMac), "AbMac");
    EXPECT_STREQ(pimOpName(PimOp::WrReg), "WrReg");
    for (unsigned i = 1; i <= NumPimOps; i++) {
        bool load = op(i) == PimOp::RdGb || op(i) == PimOp::RdReg;
        EXPECT_EQ(pimOpIsLoad(op(i)), load) << pimOpName(op(i));
    }
}

/** makeReadCmd/makeWriteCmd return the operation's command. */
TEST(PimCmd, MakeCommand)
{
    for (unsigned i = 1; i <= NumPimOps; i++) {
        auto req = pimRequest(op(i));
        MemCmd cmd = pimOpIsLoad(op(i)) ? Packet::makeReadCmd(req)
                                        : Packet::makeWriteCmd(req);
        EXPECT_EQ(cmd.toInt(), MemCmd::pimReqCmd(op(i)));
        EXPECT_EQ(cmd.toString(), pimName(op(i)) + "Req");
    }
}

/**
 * Each request and response pair has the PIM command attributes: load-like
 * operations read, the others write, every request needs a response, and
 * none is a cache, coherence or prefetch command.
 */
TEST(PimCmd, Attributes)
{
    for (unsigned i = 1; i <= NumPimOps; i++) {
        bool load = pimOpIsLoad(op(i));
        MemCmd req(MemCmd::pimReqCmd(op(i)));
        SCOPED_TRACE(req.toString());
        EXPECT_TRUE(req.isPim());
        EXPECT_TRUE(req.isRequest());
        EXPECT_FALSE(req.isResponse());
        EXPECT_TRUE(req.needsResponse());
        EXPECT_EQ(req.isRead(), load);
        EXPECT_EQ(req.isWrite(), !load);
        EXPECT_EQ(req.hasData(), !load);
        EXPECT_FALSE(req.needsWritable());
        EXPECT_FALSE(req.isInvalidate());
        EXPECT_FALSE(req.fromCache());
        EXPECT_FALSE(req.isEviction());
        EXPECT_FALSE(req.isClean());
        EXPECT_FALSE(req.isLLSC());
        EXPECT_FALSE(req.isLockedRMW());
        EXPECT_FALSE(req.isPrefetch());
        EXPECT_FALSE(req.isDemand());

        MemCmd resp(req.responseCommand());
        EXPECT_EQ(resp.toInt(), req.toInt() + 1);
        EXPECT_EQ(resp.toString(), pimName(op(i)) + "Resp");
        EXPECT_TRUE(resp.isPim());
        EXPECT_TRUE(resp.isResponse());
        EXPECT_FALSE(resp.isRequest());
        EXPECT_FALSE(resp.needsResponse());
        EXPECT_EQ(resp.isRead(), load);
        EXPECT_EQ(resp.isWrite(), !load);
        EXPECT_EQ(resp.hasData(), load);
        EXPECT_FALSE(resp.isInvalidate());
        EXPECT_EQ(resp.responseCommand(), MemCmd::InvalidCmd);
    }
    // Commands that existed before are not PIM.
    for (int c = 0; c < MemCmd::PimSbEwaddReq; c++) {
        EXPECT_FALSE(MemCmd(c).isPim()) << MemCmd(c).toString();
    }
    EXPECT_EQ(MemCmd::NUM_MEM_CMDS, MemCmd::TlbiExtSync + 1 + 2 * NumPimOps);
}

/** A malformed PIM request is a decoder bug: panic. */
TEST(PimCmd, MalformedRequestsPanic)
{
    // The other direction.
    EXPECT_THROW(Packet::makeWriteCmd(pimRequest(PimOp::RdGb)),
                 GTestException);
    EXPECT_THROW(Packet::makeReadCmd(pimRequest(PimOp::AbMac)),
                 GTestException);
    // Without UNCACHEABLE or STRICT_ORDER.
    auto no_uc = std::make_shared<Request>(
        Base, Size,
        Request::PIM | Request::STRICT_ORDER |
            Request::pimOpFlags(PimOp::WrGb),
        0);
    EXPECT_THROW(Packet::makeWriteCmd(no_uc), GTestException);
    auto no_so = std::make_shared<Request>(
        Base, Size,
        Request::PIM | Request::UNCACHEABLE |
            Request::pimOpFlags(PimOp::RdReg),
        0);
    EXPECT_THROW(Packet::makeReadCmd(no_so), GTestException);
    // With LLSC.
    EXPECT_THROW(Packet::makeReadCmd(pimRequest(PimOp::RdGb, Request::LLSC)),
                 GTestException);
    EXPECT_THROW(
        Packet::makeWriteCmd(pimRequest(PimOp::WrGb, Request::LLSC)),
        GTestException);
    // A masked write.
    auto masked = pimRequest(PimOp::WrGb);
    std::vector<bool> be(Size, true);
    be[3] = false;
    masked->setByteEnable(be);
    EXPECT_THROW(Packet::makeWriteCmd(masked), GTestException);
    // PIM without an operation.
    auto no_op = std::make_shared<Request>(
        Base, Size,
        Request::PIM | Request::UNCACHEABLE | Request::STRICT_ORDER, 0);
    EXPECT_THROW(Packet::makeWriteCmd(no_op), GTestException);
}

/** Functional accesses never read or write a queued PIM packet. */
TEST(PimCmd, FunctionalAccessIgnoresPimPayload)
{
    std::vector<uint8_t> payload(Size), probe(Size, 0x55);
    for (unsigned i = 0; i < Size; i++)
        payload[i] = i;

    auto pim_req = pimRequest(PimOp::WrGb);
    Packet pim(pim_req, Packet::makeWriteCmd(pim_req));
    pim.dataStatic(payload.data());

    // A functional read overlapping the PIM write is not served from it.
    auto rd_req = std::make_shared<Request>(Base, Size, 0, 0);
    Packet rd(rd_req, MemCmd::ReadReq);
    rd.dataStatic(probe.data());
    EXPECT_FALSE(rd.trySatisfyFunctional(&pim));
    for (unsigned i = 0; i < Size; i++)
        EXPECT_EQ(probe[i], 0x55);

    // A functional write leaves the PIM payload unchanged.
    auto wr_req = std::make_shared<Request>(Base, Size, 0, 0);
    Packet wr(wr_req, MemCmd::WriteReq);
    wr.dataStatic(probe.data());
    EXPECT_FALSE(wr.trySatisfyFunctional(&pim));
    for (unsigned i = 0; i < Size; i++)
        EXPECT_EQ(payload[i], i);

    // A plain write still serves the functional read.
    std::vector<uint8_t> plain_data(Size, 0xab);
    auto plain_req = std::make_shared<Request>(Base, Size, 0, 0);
    Packet plain(plain_req, MemCmd::WriteReq);
    plain.dataStatic(plain_data.data());
    EXPECT_TRUE(rd.trySatisfyFunctional(&plain));
    for (unsigned i = 0; i < Size; i++)
        EXPECT_EQ(probe[i], 0xab);
}
