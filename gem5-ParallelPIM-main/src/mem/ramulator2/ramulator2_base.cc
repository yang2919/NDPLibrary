/*
 * Derived from Ramulator 2.1, resources/gem5_wrappers/ramulator2_base.cc at
 * upstream commit c6c7a5d (ext/ramulator2), hardened for this gem5 tree.
 *
 * MIT License
 *
 * Copyright (c) 2023 SAFARI Research Group at ETH Zurich and Carnegie Mellon
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mem/ramulator2/ramulator2_base.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>

#include "base/callback.hh"
#include "base/logging.hh"
#include "base/output.hh"
#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/Ramulator2.hh"
#include "sim/system.hh"

// gem5 defines warn as a macro; protect Ramulator 2.1's headers from it.
#pragma push_macro("warn")
#undef warn

#include "ramulator/base/base.h"
#include "ramulator/base/config.h"
#include "ramulator/base/config_node.h"
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
                               size_t num_ports,
                               Tick frontend_latency,
                               Tick backend_latency,
                               Tick atomic_read_latency) :
    AbstractMemory(p),
    ramulator_config(_ramulator_config),
    ramulator2_frontend(nullptr), ramulator2_memorysystem(nullptr),
    ramulator2_finalized(false),
    startTick(0),
    nbrOutstandingReads(0), nbrOutstandingWrites(0),
    frontendLatency(frontend_latency), backendLatency(backend_latency),
    atomicReadLatency(atomic_read_latency),
    dramClockPeriod(0),
    tickEvent([this]{ tick(); }, name())
{
    for (size_t i = 0; i < num_ports; ++i) {
        portStates.push_back(std::make_unique<PortState>(
            *this, static_cast<PortID>(i)));
    }

    registerExitCallback([this]() {
        writeRamulatorStats(statsFileName(""), StatsWriteMode::Final);
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
    Ramulator::ConfigNode config =
        Ramulator::Config::parse_config_string(ramulator_config);
    ramulator2_frontend = Ramulator::Factory::create_frontend(config);
    ramulator2_memorysystem = Ramulator::Factory::create_memory_system(config);

    ramulator2_frontend->connect_memory_system(ramulator2_memorysystem);
    ramulator2_memorysystem->connect_frontend(ramulator2_frontend);

    // tCK in picoseconds is an integer; round instead of truncating a
    // float round trip, which can be one picosecond short.
    dramClockPeriod = std::llround(
        ramulator2_memorysystem->get_tCK() * 1000.0f) * sim_clock::as_int::ps;
    fatal_if(dramClockPeriod == 0,
             "%s: Ramulator 2.1 reports a DRAM clock period of %f ns, which "
             "rounds to 0 ps.\n", name(), ramulator2_memorysystem->get_tCK());
    inform("%s: Ramulator 2.1 DRAM clock period %llu ps, %d bytes per "
           "transaction\n", name(),
           (unsigned long long)(dramClockPeriod / sim_clock::as_int::ps),
           ramulator2_memorysystem->get_tx_bytes());
}

void
Ramulator2Base::startup()
{
    startTick = curTick();

    // The first DRAM cycle is at the first clock edge, in SE and FS alike.
    // Upstream delays it to tick 13121004000177 in FS; RISC-V Linux boots
    // without that delay.
    schedule(tickEvent, clockEdge());
}

void
Ramulator2Base::resetStats()
{
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

    writeRamulatorStats(statsFileName("." + std::to_string(curTick())),
                        StatsWriteMode::Snapshot);
}

std::string
Ramulator2Base::statsFileName(const std::string& suffix) const
{
    // One file per instance, named by the object's path, so that several
    // Ramulator 2.1 memories in one run do not overwrite each other.
    return simout.resolve(name() + ".ramulator_stats" + suffix + ".yaml");
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

    const std::string text = ramulatorStatsText();
    if (mode == StatsWriteMode::Snapshot)
        updateStatsBridge(text);

    std::ofstream ofs(path);
    if (!ofs) {
        fatal("Ramulator2 failed to open stats file %s\n", path.c_str());
    }

    ofs << text;
    ofs.flush();

    if (!ofs) {
        fatal("Ramulator2 failed to write stats file %s\n", path.c_str());
    }
}

std::string
Ramulator2Base::ramulatorStatsText() const
{
    std::ostringstream os;
    ramulator2_frontend->print_stats(os);
    ramulator2_memorysystem->print_stats(os);
    return os.str();
}

namespace
{

/** A YAML scalar that is a number, as Ramulator 2.1 prints statistics. */
bool
parseNumber(const std::string& text, double& value)
{
    if (text.empty())
        return false;
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end != text.c_str() && *end == '\0';
}

/** Whether a statistics subtree holds any numeric leaf. */
bool
hasNumericLeaf(const Ramulator::ConfigNode& node)
{
    double unused;
    if (node.is_scalar())
        return parseNumber(node.scalar(), unused);
    if (node.is_sequence()) {
        for (const auto& item : node.seq()) {
            if (hasNumericLeaf(item))
                return true;
        }
        return false;
    }
    if (node.is_map()) {
        for (const auto& [key, child] : node.map()) {
            if (key != "impl" && key != "id" && hasNumericLeaf(child))
                return true;
        }
    }
    return false;
}

/** A valid gem5 statistics name for a Ramulator 2.1 key. */
std::string
statName(const std::string& key)
{
    std::string name = key;
    for (auto& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)))
            c = '_';
    }
    if (name.empty() || std::isdigit(static_cast<unsigned char>(name[0])))
        name = "_" + name;
    return name;
}

} // anonymous namespace

void
Ramulator2Base::regStats()
{
    AbstractMemory::regStats();

    if (!ramulator2_memorysystem)
        return;

    // The statistics tree, as the YAML files print it. Its shape is fixed
    // once Ramulator 2.1 is built; only the values change.
    const Ramulator::ConfigNode tree =
        Ramulator::Config::parse_config_string(ramulatorStatsText());
    bridgeStats(tree, this, "", true);
}

void
Ramulator2Base::bridgeStats(const Ramulator::ConfigNode& node,
                            statistics::Group* group,
                            const std::string& path, bool create)
{
    if (!node.is_map())
        return;

    auto subgroup = [&](statistics::Group* parent,
                        const std::string& name) -> statistics::Group* {
        if (!create)
            return nullptr;
        statsBridge.groups.push_back(
            std::make_unique<statistics::Group>(parent, name.c_str()));
        return statsBridge.groups.back().get();
    };

    auto leaf = [&](statistics::Group* parent, const std::string& name,
                    const std::string& leaf_path,
                    const Ramulator::ConfigNode& value_node) {
        double value;
        if (!value_node.is_scalar() ||
            !parseNumber(value_node.scalar(), value)) {
            return;
        }
        if (create) {
            const size_t i = statsBridge.cache.size();
            statsBridge.cache.push_back(value);
            statsBridge.index[leaf_path] = i;
            const std::string desc = "Ramulator 2.1 statistic " + leaf_path;
            statsBridge.values.push_back(std::make_unique<statistics::Value>(
                parent, name.c_str(), statistics::units::Unspecified::get(),
                desc.c_str()));
            statsBridge.values.back()->functor(
                [this, i]() -> statistics::Result {
                    return statsBridge.cache[i];
                });
        } else {
            auto it = statsBridge.index.find(leaf_path);
            if (it != statsBridge.index.end())
                statsBridge.cache[it->second] = value;
        }
    };

    for (const auto& [key, child] : node.map()) {
        if (key == "impl" || key == "id")
            continue;
        const std::string child_path = path.empty() ? key : path + "." + key;
        // Controllers are the channels: their groups sit next to the
        // memory system's, as channel0, channel1, ...
        const bool channels = path == "memory_system" && key == "controller";

        if (child.is_scalar()) {
            leaf(group, statName(key), child_path, child);
        } else if (child.is_map()) {
            if (hasNumericLeaf(child)) {
                bridgeStats(child,
                            subgroup(channels ? this : group,
                                     channels ? "channel0" : statName(key)),
                            channels ? child_path + ".0" : child_path,
                            create);
            }
        } else if (child.is_sequence()) {
            const auto& items = child.seq();
            for (size_t i = 0; i < items.size(); ++i) {
                const std::string item_path =
                    child_path + "." + std::to_string(i);
                const std::string index = std::to_string(i);
                if (items[i].is_map()) {
                    if (!hasNumericLeaf(items[i]))
                        continue;
                    bridgeStats(items[i],
                                subgroup(channels ? this : group,
                                         channels ? "channel" + index :
                                         statName(key) + index),
                                item_path, create);
                } else {
                    leaf(group, statName(key) + "_" + index, item_path,
                         items[i]);
                }
            }
        }
    }
}

void
Ramulator2Base::updateStatsBridge(const std::string& stats_text)
{
    if (statsBridge.values.empty())
        return;
    bridgeStats(Ramulator::Config::parse_config_string(stats_text), nullptr,
                "", false);
}

void
Ramulator2Base::sendResponse(PortID port_id)
{
    auto& state = *portStates.at(port_id);
    assert(!state.retryResp);
    assert(!state.responseQueue.empty());

    DPRINTF(Ramulator2, "Attempting to send response\n");

    bool success = getMemoryPort(port_id).sendTimingResp(
        state.responseQueue.front().second);
    if (success) {
        state.responseQueue.pop_front();

        DPRINTF(Ramulator2, "Have %d read, %d write, %d responses "
                "outstanding\n", nbrOutstandingReads, nbrOutstandingWrites,
                state.responseQueue.size());

        if (!state.responseQueue.empty() &&
            !state.sendResponseEvent.scheduled()) {
            schedule(state.sendResponseEvent,
                     std::max(curTick(), state.responseQueue.front().first));
        }

        if (drained())
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
        // The parts of a split packet that are not sent yet. Without them,
        // the completion of a write's first part could report the memory
        // drained while the rest still has to go to Ramulator 2.1.
        if (state->pendingSplit) {
            const Addr tx = ramulator2_memorysystem->get_tx_bytes();
            outstanding +=
                (state->pendingEnd - state->pendingNext + tx - 1) / tx;
        }
    }
    return outstanding;
}

void
Ramulator2Base::tick()
{
    // Only tick when it's timing mode
    if (system()->isTimingMode()) {
        ramulator2_memorysystem->tick();
        cycle();

        // Feed the waiting parts of split packets, then, if a port is
        // waiting for a retry and has nothing left to feed, send it.
        for (size_t i = 0; i < portStates.size(); ++i) {
            PortID port_id = static_cast<PortID>(i);
            auto& state = *portStates[i];
            feedPendingSplit(port_id);
            if (state.retryReq && !state.pendingSplit) {
                state.retryReq = false;
                getMemoryPort(port_id).sendRetryReq();
            }
        }
    }

    schedule(tickEvent, curTick() + dramClockPeriod);
}

Tick
Ramulator2Base::recvAtomic(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");
    // PIM requests need a PIM memory (PimRamulator2).
    fatal_if(pkt->isPim(), "%s: a PIM request (%s) reached a memory without "
             "PIM support; use a PimMemory", name(), pkt->print());

    access(pkt);
    // As MemCtrl::recvAtomicLogic: a closed-page latency for a response
    // with data, 0 otherwise.
    return pkt->hasData() ? atomicReadLatency : 0;
}

void
Ramulator2Base::recvFunctional(PacketPtr pkt)
{
    pkt->pushLabel(name());
    functionalAccess(pkt);

    for (auto& state : portStates) {
        for (auto i = state->responseQueue.begin();
             i != state->responseQueue.end(); ++i) {
            pkt->trySatisfyFunctional(i->second);
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
    // PIM requests need a PIM memory (PimRamulator2).
    fatal_if(pkt->isPim(), "%s: a PIM request (%s) reached a memory without "
             "PIM support; use a PimMemory", name(), pkt->print());

    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "Should only see read and writes at memory controller, "
             "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());

    // A port that is waiting for a retry, or still feeding a split packet,
    // refuses new packets; the retry follows from tick().
    auto& state = *portStates.at(port_id);
    if (state.retryReq || state.pendingSplit) {
        state.retryReq = true;
        return false;
    }

    const Addr tx = ramulator2_memorysystem->get_tx_bytes();
    const Addr base = pkt->getAddr();
    const Addr end = base + pkt->getSize();
    // As native gem5, atomic memory operations take the write path.
    const bool is_write = pkt->isWrite();
    unsigned chunks = 0;
    for (Addr a = base; a < end; a = (a | (tx - 1)) + 1)
        ++chunks;

    auto split = std::make_shared<SplitPacket>(SplitPacket{pkt, chunks});
    if (!sendChunk(base, end, is_write, port_id, split)) {
        state.retryReq = true;
        return false;
    }

    state.pendingNext = (base | (tx - 1)) + 1;
    state.pendingEnd = end;
    state.pendingWrite = is_write;
    if (state.pendingNext < end)
        state.pendingSplit = split;

    if (is_write) {
        accessAndRespond(pkt, port_id, frontendLatency);
    } else {
        // Counted as outstanding until its last part has completed and
        // its response has been queued.
        ++nbrOutstandingReads;
    }
    feedPendingSplit(port_id);
    return true;
}

bool
Ramulator2Base::sendChunk(Addr addr, Addr end, bool is_write, PortID port_id,
                          std::shared_ptr<SplitPacket> split)
{
    const Addr tx = ramulator2_memorysystem->get_tx_bytes();
    const int size = std::min((addr | (tx - 1)) + 1, end) - addr;
    // Decode from the start of the memory's range, as native gem5's
    // decodePacket does.
    const Addr ctrl_addr = getAddrRange().getOffset(addr);
    const int ingress = getIngressId(port_id);
    const uint64_t token = nextChunkToken++;
    if (is_write) {
        ++nbrOutstandingWrites;
        bool ok = ramulator2_frontend->receive_external_requests(
            1, ctrl_addr, 0, ingress,
            [this, token](Ramulator::Request& req) {
                DPRINTF(Ramulator2, "Write to %ld completed.\n", req.addr);
                chunkDone(token);
                --nbrOutstandingWrites;
                if (drained())
                    signalDrainDone();
            }, size);
        if (!ok)
            --nbrOutstandingWrites;
        else
            chunkSent(token, ctrl_addr, true, port_id, *split);
        return ok;
    }
    bool ok = ramulator2_frontend->receive_external_requests(
        0, ctrl_addr, 0, ingress,
        [this, port_id, split, token](Ramulator::Request& req) {
            DPRINTF(Ramulator2, "Read to %ld completed.\n", req.addr);
            chunkDone(token);
            if (--split->left == 0) {
                --nbrOutstandingReads;
                accessAndRespond(split->pkt, port_id,
                                 frontendLatency + backendLatency);
            }
        }, size);
    if (ok)
        chunkSent(token, ctrl_addr, false, port_id, *split);
    return ok;
}

void
Ramulator2Base::feedPendingSplit(PortID port_id)
{
    auto& state = *portStates.at(port_id);
    if (!state.pendingSplit)
        return;
    const Addr tx = ramulator2_memorysystem->get_tx_bytes();
    while (state.pendingNext < state.pendingEnd) {
        if (!sendChunk(state.pendingNext, state.pendingEnd,
                       state.pendingWrite, port_id, state.pendingSplit))
            return;
        state.pendingNext = (state.pendingNext | (tx - 1)) + 1;
    }
    state.pendingSplit.reset();
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
Ramulator2Base::accessAndRespond(PacketPtr pkt, PortID port_id,
                                 Tick static_latency)
{
    DPRINTF(Ramulator2, "Access for address %lld\n", pkt->getAddr());

    bool needsResponse = pkt->needsResponse();

    access(pkt);

    // turn packet around to go back to requestor if response expected
    if (needsResponse) {
        // access already turned the packet into a response
        assert(pkt->isResponse());
        queueResponse(pkt, port_id, static_latency);
    } else {
        // queue the packet for deletion
        portStates.at(port_id)->pendingDelete.reset(pkt);
    }
}

void
Ramulator2Base::queueResponse(PacketPtr pkt, PortID port_id,
                              Tick static_latency)
{
    {
        Tick time = curTick() + static_latency + pkt->headerDelay +
                    pkt->payloadDelay;
        // Here we reset the timing of the packet before sending it out.
        pkt->headerDelay = pkt->payloadDelay = 0;

        DPRINTF(Ramulator2, "Queuing response for address %lld\n",
                pkt->getAddr());

        // Insert by due time; each packet keeps its own delay.
        auto& state = *portStates.at(port_id);
        auto it = state.responseQueue.end();
        while (it != state.responseQueue.begin() &&
               std::prev(it)->first > time) {
            --it;
        }
        bool new_head = it == state.responseQueue.begin();
        state.responseQueue.insert(it, {time, pkt});

        // if we are not already waiting for a retry, schedule an event for
        // the earliest response
        if (!state.retryResp) {
            if (!state.sendResponseEvent.scheduled())
                schedule(state.sendResponseEvent, time);
            else if (new_head)
                reschedule(state.sendResponseEvent, time);
        }
    }
}

DrainState
Ramulator2Base::drain()
{
    // check our outstanding requests, including the unsent parts of split
    // packets, and if any they need to drain
    return drained() ? DrainState::Drained : DrainState::Draining;
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
