#include "mem/ramulator2/ramulator2_base.hh"

#include <fstream>

#include "base/callback.hh"
#include "base/output.hh"
#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/Ramulator2.hh"
#include "sim/full_system.hh"
#include "sim/system.hh"

// gem5 defines warn as a macro — protect Ramulator headers
#pragma push_macro("warn")
#undef warn

#include "ramulator/base/base.h"
#include "ramulator/base/config.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

#pragma pop_macro("warn")

namespace gem5
{

namespace memory
{

Ramulator2Base::Ramulator2Base(const AbstractMemoryParams& p,
                               const std::string& _ramulator_config,
                               size_t num_ports) :
    AbstractMemory(p),
    ramulator_config(_ramulator_config),
    ramulator2_frontend(nullptr), ramulator2_memorysystem(nullptr),
    ramulator2_finalized(false),
    startTick(0),
    nbrOutstandingReads(0), nbrOutstandingWrites(0),
    tickEvent([this]{ tick(); }, name())
{
    for (size_t i = 0; i < num_ports; ++i) {
        portStates.push_back(std::make_unique<PortState>(
            *this, static_cast<PortID>(i)));
    }

    registerExitCallback([this]() {
        writeRamulatorStats(
            simout.resolve("ramulator_stats.yaml"),
            StatsWriteMode::Final);
    });
}

Ramulator2Base::~Ramulator2Base()
{
    delete ramulator2_frontend;
    delete ramulator2_memorysystem;
}

void
Ramulator2Base::initRamulator()
{
    Ramulator::ConfigNode config = Ramulator::Config::parse_config_string(ramulator_config);
    ramulator2_frontend = Ramulator::Factory::create_frontend(config);
    ramulator2_memorysystem = Ramulator::Factory::create_memory_system(config);

    ramulator2_frontend->connect_memory_system(ramulator2_memorysystem);
    ramulator2_memorysystem->connect_frontend(ramulator2_frontend);

    // gem5 packet size is passed through to Ramulator; the memory system
    // validates that each request fits in one DRAM transaction.
}

void
Ramulator2Base::startup()
{
    startTick = curTick();

    if (FullSystem) {
        // This delayed first tick avoids early FS boot issues observed by the
        // original wrapper, while still supporting checkpoint restores after
        // that point. credits: @sangjae4309
        // please check https://github.com/CMU-SAFARI/ramulator2/pull/79 and
        // https://github.com/sangjae4309/gem5-ramulator2/issues/5 for details
        constexpr Tick fsBootWorkaroundTick = 13121004000177;
        schedule(tickEvent,
            curTick() < fsBootWorkaroundTick ?
            fsBootWorkaroundTick : clockEdge());
    } else {
        schedule(tickEvent, clockEdge());
    }
}

void
Ramulator2Base::resetStats() {
    ClockedObject::resetStats();

    if (ramulator2_frontend)
        ramulator2_frontend->reset_stats_recursive();
    if (ramulator2_memorysystem)
        ramulator2_memorysystem->reset_stats_recursive();
}

void
Ramulator2Base::preDumpStats()
{
    ClockedObject::preDumpStats();

    writeRamulatorStats(simout.resolve(
        "ramulator_stats." + std::to_string(curTick()) + ".yaml"),
        StatsWriteMode::Snapshot);
}

void
Ramulator2Base::writeRamulatorStats(const std::string& path,
                                    StatsWriteMode mode)
{
    if (!ramulator2_frontend || !ramulator2_memorysystem)
        return;

    if (mode == StatsWriteMode::Final) {
        if (!ramulator2_finalized) {
            ramulator2_frontend->finalize();
            ramulator2_memorysystem->finalize();
            ramulator2_finalized = true;
        }
    } else {
        ramulator2_frontend->update_stats_recursive();
        ramulator2_memorysystem->update_stats_recursive();
    }

    std::ofstream ofs(path);
    if (!ofs) {
        fatal("Ramulator2 failed to open stats file %s\n", path.c_str());
    }

    ramulator2_frontend->print_stats(ofs);
    ramulator2_memorysystem->print_stats(ofs);
    ofs.flush();

    if (!ofs) {
        fatal("Ramulator2 failed to write stats file %s\n", path.c_str());
    }
}

void
Ramulator2Base::sendResponse(PortID port_id)
{
    auto& state = *portStates.at(port_id);
    assert(!state.retryResp);
    assert(!state.responseQueue.empty());

    DPRINTF(Ramulator2, "Attempting to send response\n");

    bool success = getMemoryPort(port_id).sendTimingResp(
        state.responseQueue.front());
    if (success) {
        state.responseQueue.pop_front();

        DPRINTF(Ramulator2, "Have %d read, %d write, %d responses outstanding\n",
                nbrOutstandingReads, nbrOutstandingWrites,
                state.responseQueue.size());

        if (!state.responseQueue.empty() &&
            !state.sendResponseEvent.scheduled()) {
            schedule(state.sendResponseEvent, curTick());
        }

        if (nbrOutstanding() == 0)
            signalDrainDone();
    } else {
        state.retryResp = true;

        DPRINTF(Ramulator2, "Waiting for response retry\n");

        assert(!state.sendResponseEvent.scheduled());
    }
}

unsigned int
Ramulator2Base::nbrOutstanding() const
{
    unsigned int outstanding = nbrOutstandingReads + nbrOutstandingWrites;
    for (const auto& state : portStates) {
        outstanding += state->responseQueue.size();
    }
    return outstanding;
}

void
Ramulator2Base::tick()
{
    // Only tick when it's timing mode
    if (system()->isTimingMode()) {
        ramulator2_memorysystem->tick();

        // is the connected port waiting for a retry, if so check the
        // state and send a retry if conditions have changed
        for (size_t i = 0; i < portStates.size(); ++i) {
            PortID port_id = static_cast<PortID>(i);
            auto& state = *portStates[i];
            if (state.retryReq) {
                state.retryReq = false;
                getMemoryPort(port_id).sendRetryReq();
            }
        }
    }

    schedule(tickEvent,
        curTick() + ramulator2_memorysystem->get_tCK() * sim_clock::as_float::ns);
}

Tick
Ramulator2Base::recvAtomic(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");

    access(pkt);
    return 50000;   // Arbitrary latency of 50ns
}

void
Ramulator2Base::recvFunctional(PacketPtr pkt)
{
    pkt->pushLabel(name());
    functionalAccess(pkt);

    for (auto& state : portStates) {
        for (auto i = state->responseQueue.begin();
             i != state->responseQueue.end(); ++i) {
            pkt->trySatisfyFunctional(*i);
        }
    }

    pkt->popLabel();
}

bool
Ramulator2Base::recvTimingReq(PacketPtr pkt, PortID port_id)
{
    DPRINTF(Ramulator2, "recvTimingReq: request %s addr %#x size %d\n",
            pkt->cmdString(), pkt->getAddr(), pkt->getSize());

    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");

    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "Should only see read and writes at memory controller, "
             "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());

    // we should not get a new request after committing to retry the
    // current one, but unfortunately the CPU violates this rule, so
    // simply ignore it for now
    auto& state = *portStates.at(port_id);
    if (state.retryReq)
        return false;

    bool enqueue_success = false;
    const int ingress_id = getIngressId(port_id);
    if (pkt->isRead())
    {
        // Generate ramulator READ request and try to send to ramulator's memory system
        enqueue_success = ramulator2_frontend->
            receive_external_requests(0, pkt->getAddr(), 0, ingress_id,
            [this, port_id](Ramulator::Request& req) {
                DPRINTF(Ramulator2, "Read to %ld completed.\n", req.addr);
                auto& pkt_q = outstandingReads.find(req.addr)->second;
                PacketPtr pkt = pkt_q.front();
                pkt_q.pop_front();
                if (!pkt_q.size())
                    outstandingReads.erase(req.addr);

                --nbrOutstandingReads;

                accessAndRespond(pkt, port_id);
            },
            pkt->getSize());

        if (enqueue_success)
        {
            outstandingReads[pkt->getAddr()].push_back(pkt);

            // we count a transaction as outstanding until it has left the
            // queue in the controller, and the response has been sent
            // back, note that this will differ for reads and writes
            ++nbrOutstandingReads;
        }
        else
        {
            state.retryReq = true;
        }
    } else if (pkt->isWrite()) {
        enqueue_success = ramulator2_frontend->
            receive_external_requests(1, pkt->getAddr(), 0, ingress_id,
            [this](Ramulator::Request& req) {
                DPRINTF(Ramulator2, "Write to %ld completed.\n", req.addr);
                --nbrOutstandingWrites;
                if (nbrOutstanding() == 0)
                    signalDrainDone();
            },
            pkt->getSize());

        if (enqueue_success)
        {
            accessAndRespond(pkt, port_id);
            ++nbrOutstandingWrites;
        }
        else
        {
            state.retryReq = true;
        }
    } else {
        // keep it simple and just respond if necessary
        accessAndRespond(pkt, port_id);
        return true;
    }

    return enqueue_success;
}

void
Ramulator2Base::recvRespRetry(PortID port_id)
{
    DPRINTF(Ramulator2, "Retrying\n");

    auto& state = *portStates.at(port_id);
    assert(state.retryResp);
    state.retryResp = false;
    sendResponse(port_id);
}

void
Ramulator2Base::accessAndRespond(PacketPtr pkt, PortID port_id)
{
    DPRINTF(Ramulator2, "Access for address %lld\n", pkt->getAddr());

    bool needsResponse = pkt->needsResponse();

    access(pkt);

    // turn packet around to go back to requestor if response expected
    if (needsResponse) {
        // access already turned the packet into a response
        assert(pkt->isResponse());

        // Assume frontend latency = 0
        Tick time = curTick() + pkt->headerDelay + pkt->payloadDelay;
        // Here we reset the timing of the packet before sending it out.
        pkt->headerDelay = pkt->payloadDelay = 0;

        DPRINTF(Ramulator2, "Queuing response for address %lld\n",
                pkt->getAddr());

        // queue it to be sent back
        auto& state = *portStates.at(port_id);
        state.responseQueue.push_back(pkt);

        // if we are not already waiting for a retry, or are scheduled
        // to send a response, schedule an event
        if (!state.retryResp && !state.sendResponseEvent.scheduled())
            schedule(state.sendResponseEvent, time);
    } else {
        // queue the packet for deletion
        portStates.at(port_id)->pendingDelete.reset(pkt);
    }
}

DrainState
Ramulator2Base::drain()
{
    // check our outstanding reads and writes and if any they need to drain
    return nbrOutstanding() != 0 ? DrainState::Draining : DrainState::Drained;
}

Ramulator2Base::MemorySystemPort::MemorySystemPort(
        const std::string& _name, Ramulator2Base& _ramulator2,
        PortID _port_id)
    : ResponsePort(_name), ramulator2(_ramulator2), portId(_port_id)
{ }

AddrRangeList
Ramulator2Base::MemorySystemPort::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(ramulator2.getPortRange(portId));
    return ranges;
}

Ramulator2Base::PortState::PortState(Ramulator2Base& ramulator2,
                                     PortID port_id)
    : sendResponseEvent(
          [&ramulator2, port_id]{ ramulator2.sendResponse(port_id); },
          ramulator2.name() + ".sendResponse" + std::to_string(port_id))
{ }

} // namespace memory
} // namespace gem5
