/*
 * The unloaded latency U of a PIM request; see pim_unloaded.hh.
 */

#include "mem/ramulator2/pim/pim_unloaded.hh"

#include <algorithm>
#include <cassert>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>

namespace gem5
{

namespace pim
{

using Ramulator::PimClass;

namespace
{

/** One node of a channel's tree, as Ramulator 2.1's DRAMNode. */
struct Node
{
    int level = 0;
    int id = 0;
    std::vector<int64_t> ready;
    std::vector<std::deque<int64_t>> hist;
    std::vector<std::deque<int64_t>> shared;
    std::vector<std::unique_ptr<Node>> kids;
    Node *parent = nullptr;
    // Banks only.
    enum class State
    {
        Closed,
        Activating,
        Opened
    } state = State::Closed;
    int row = -1;
};

int
nameIndex(const std::vector<std::string> &names, const std::string &n)
{
    auto it = std::find(names.begin(), names.end(), n);
    return it == names.end() ? -1 : int(it - names.begin());
}

} // anonymous namespace

PimUnloaded::PimUnloaded(const Ramulator::PimTimingInfo &info) : t(info)
{
    levels = int(t.level_names.size());
    rowLevel = nameIndex(t.level_names, "Row");
    colLevel = nameIndex(t.level_names, "Column");
    bankLevel = nameIndex(t.level_names, "Bank");
    unitLevel = nameIndex(t.level_names, "Rank");
    if (unitLevel < 0)
        unitLevel = nameIndex(t.level_names, "PseudoChannel");
    if (unitLevel < 0)
        unitLevel = 0;
    if (rowLevel < 0 || colLevel < 0 || bankLevel < 0)
        throw std::runtime_error("PimUnloaded: no bank, row or column level");
    const int ncmd = int(t.command_names.size());
    cons.assign(levels, std::vector<std::vector<Ramulator::PimTimingEntry>>(
                            ncmd));
    window.assign(levels, std::vector<int>(ncmd, 0));
    groupWindow.assign(levels, {});
    for (const auto &e : t.entries) {
        cons[e.level][e.cmd].push_back(e);
        window[e.level][e.cmd] = std::max(window[e.level][e.cmd], e.window);
        if (e.history_group >= 0) {
            auto &g = groupWindow[e.level];
            if (int(g.size()) <= e.history_group)
                g.resize(e.history_group + 1, 0);
            g[e.history_group] = std::max(g[e.history_group], e.window);
        }
    }
    lpddr = t.standard.find("LPDDR") != std::string::npos;
    hbm34 = t.standard.find("HBM3") != std::string::npos ||
            t.standard.find("HBM4") != std::string::npos;
    cmdCas = nameIndex(t.command_names, "CAS");
    isAccess.assign(ncmd, false);
    isPre.assign(ncmd, false);
    for (int c = 0; c < ncmd; c++) {
        const std::string &n = t.command_names[c];
        isPre[c] = n.rfind("PRE", 0) == 0;
        // LPDDR controllers synchronize the write clock before data-carrying
        // commands; of the PIM commands, the non-bank ones.
        isAccess[c] = n == "PIM_GB_RD" || n == "PIM_GB_WR" ||
                      n == "PIM_REG_RD" || n == "PIM_REG_WR";
    }
    if (lpddr && cmdCas < 0)
        lpddr = false;
}

int
PimUnloaded::timing(const std::string &name) const
{
    const int i = nameIndex(t.timing_names, name);
    return i < 0 ? -1 : t.timing_vals[i];
}

int64_t
PimUnloaded::clocks(PimClass cls, const std::vector<int> &job_addr,
                    int ncols, int64_t first) const
{
    const int ncmd = int(t.command_names.size());

    // The channel's tree (DRAMNode's constructor).
    std::function<std::unique_ptr<Node>(int, int, Node *)> make =
        [&](int level, int id, Node *parent) {
            auto n = std::make_unique<Node>();
            n->level = level;
            n->id = id;
            n->parent = parent;
            n->ready.assign(ncmd, -1);
            n->hist.resize(ncmd);
            for (int c = 0; c < ncmd; c++) {
                if (window[level][c])
                    n->hist[c].assign(window[level][c], -1);
            }
            for (int w : groupWindow[level])
                n->shared.emplace_back(std::deque<int64_t>(w, -1));
            const int next = level + 1;
            if (next == rowLevel)
                return n;
            const int size = t.level_sizes[next];
            for (int i = 0; i < size; i++)
                n->kids.push_back(make(next, i, n.get()));
            return n;
        };
    std::unique_ptr<Node> root = make(0, 0, nullptr);

    // DRAMNode::update_timing.
    std::function<void(Node &, int, const std::vector<int> &, int64_t)>
        update = [&](Node &n, int cmd, const std::vector<int> &a,
                     int64_t clk) {
            const auto &es = cons[n.level][cmd];
            if (n.id != a[n.level] && a[n.level] != -1) {
                for (const auto &e : es) {
                    if (e.sibling) {
                        n.ready[e.next] =
                            std::max(n.ready[e.next], clk + e.val);
                    }
                }
                return;
            }
            if (!n.hist[cmd].empty()) {
                n.hist[cmd].pop_back();
                n.hist[cmd].push_front(clk);
            }
            std::vector<bool> done(n.shared.size(), false);
            for (const auto &e : es) {
                if (e.history_group < 0 || done[e.history_group])
                    continue;
                auto &h = n.shared[e.history_group];
                h.pop_back();
                h.push_front(clk + t.command_cycles[cmd] - 1);
                done[e.history_group] = true;
            }
            for (const auto &e : es) {
                if (e.sibling)
                    continue;
                const int64_t past =
                    e.history_group >= 0
                        ? n.shared[e.history_group][e.window - 1]
                        : n.hist[cmd][e.window - 1];
                if (past < 0)
                    continue;
                n.ready[e.next] = std::max(n.ready[e.next], past + e.val);
            }
            if (n.kids.empty())
                return;
            const int child = n.level + 1;
            const int target = a[child];
            const bool sib = child < int(t.has_sibling_cons.size()) &&
                             t.has_sibling_cons[child][cmd];
            if (sib || target == -1) {
                for (auto &k : n.kids)
                    update(*k, cmd, a, clk);
            } else {
                update(*n.kids[target], cmd, a, clk);
            }
        };

    // DRAMNode::check_timing.
    std::function<bool(const Node &, int, const std::vector<int> &, int64_t)>
        check = [&](const Node &n, int cmd, const std::vector<int> &a,
                    int64_t clk) {
            if (n.ready[cmd] != -1 && clk < n.ready[cmd])
                return false;
            if (n.kids.empty())
                return true;
            const int child = a[n.level + 1];
            if (child == -1) {
                for (const auto &k : n.kids) {
                    if (!check(*k, cmd, a, clk))
                        return false;
                }
                return true;
            }
            return check(*n.kids[child], cmd, a, clk);
        };

    // DRAMDevice::check_timing and issue_command: a Pair command on both
    // banks of the pair.
    auto pairOf = [&](std::vector<int> a, int half) {
        a[bankLevel] = (a[bankLevel] & ~1) | half;
        return a;
    };
    auto checkCmd = [&](int cmd, const std::vector<int> &a, int64_t clk) {
        if (t.is_pair[cmd]) {
            return check(*root, cmd, pairOf(a, 0), clk) &&
                   check(*root, cmd, pairOf(a, 1), clk);
        }
        return check(*root, cmd, a, clk);
    };
    auto updateCmd = [&](int cmd, const std::vector<int> &a, int64_t clk) {
        if (t.is_pair[cmd]) {
            update(*root, cmd, pairOf(a, 0), clk);
            update(*root, cmd, pairOf(a, 1), clk);
        } else {
            update(*root, cmd, a, clk);
        }
    };

    auto bankNode = [&](const std::vector<int> &a) -> Node * {
        Node *n = root.get();
        for (int l = 1; l <= bankLevel; l++)
            n = n->kids.at(a[l]).get();
        return n;
    };
    auto rowOpen = [&](const std::vector<int> &a) {
        const Node *b = bankNode(a);
        return b->state == Node::State::Opened && b->row == a[rowLevel];
    };
    auto unitAddr = [&](std::vector<int> a, int col) {
        for (int l = unitLevel + 1; l < rowLevel; l++)
            a[l] = -1;
        a[colLevel] = col;
        return a;
    };
    auto pairAddr = [&](std::vector<int> a, int half) {
        a[bankLevel] = (a[bankLevel] & ~1) | half;
        return a;
    };

    std::vector<int> base = job_addr;
    base[0] = 0; // the replica has one channel
    const int ci = int(cls);
    const bool sb = Ramulator::pim_class_is_sb(cls);
    const bool ab = Ramulator::pim_class_is_ab(cls);
    const std::string &mode = t.ab_mode;
    const bool aespa = ab && mode == "aespa";

    // The job's banks (PimSequencer::submit).
    std::vector<std::vector<int>> banks;
    if (cls == PimClass::SbRd2) {
        banks = {pairAddr(base, 0), pairAddr(base, 1)};
    } else if (sb) {
        banks = {base};
    } else if (ab) {
        // Every bank of the unit, in the device's order.
        std::function<void(const Node &, std::vector<int> &)> walk =
            [&](const Node &n, std::vector<int> &a) {
                a[n.level] = n.id;
                if (n.level == bankLevel) {
                    std::vector<int> b = a;
                    b[rowLevel] = base[rowLevel];
                    b[colLevel] = base[colLevel];
                    banks.push_back(b);
                    return;
                }
                for (const auto &k : n.kids)
                    walk(*k, a);
            };
        std::vector<int> a(levels, 0);
        const Node *unit = root.get();
        if (unitLevel > 0)
            unit = unit->kids.at(base[unitLevel]).get();
        a = base;
        walk(*unit, a);
    }
    int units = 0, colsTotal = ncols;
    std::vector<int> unitCols, actOrder;
    if (aespa) {
        units = int(banks.size()) / (cls == PimClass::AbEw ? 2 : 1);
        unitCols.assign(units, 0);
        colsTotal = ncols * units;
    }
    const PimClass aespaCls = cls == PimClass::AbEw    ? PimClass::SbRd2
                              : cls == PimClass::AbRegbk ? PimClass::SbWr
                                                         : PimClass::SbRd;
    auto aespaAddr = [&](int k) {
        std::vector<int> a =
            banks[cls == PimClass::AbEw ? 2 * k : k];
        a[colLevel] = base[colLevel] + unitCols[k];
        return a;
    };
    auto unitOpen = [&](int k) {
        const std::vector<int> a = aespaAddr(k);
        return rowOpen(a) &&
               (cls != PimClass::AbEw || rowOpen(pairAddr(a, 1)));
    };

    // PimSequencer::open_step (a PREpb never occurs on an idle unit).
    auto openStep = [&](int &cmd, std::vector<int> &a) {
        for (const auto &b : banks) {
            if (rowOpen(b))
                continue;
            const Node *n = bankNode(b);
            cmd = n->state == Node::State::Activating ? t.cmd_act2
                  : t.cmd_act >= 0                    ? t.cmd_act
                                                      : t.cmd_act1;
            a = b;
            return true;
        }
        return false;
    };

    int colsIssued = 0;
    // The sequencer's next command (PimSequencer::next_command, ab_next).
    auto next = [&](int &cmd, std::vector<int> &a, int64_t clk) {
        if (ab && (mode == "ideal" || mode == "aim")) {
            bool all = true, allActivating = true;
            for (const auto &b : banks) {
                all = all && rowOpen(b);
                allActivating =
                    allActivating &&
                    bankNode(b)->state == Node::State::Activating;
            }
            if (!all) {
                cmd = t.cmd_actab >= 0 ? t.cmd_actab
                      : allActivating  ? t.cmd_actab2
                                       : t.cmd_actab1;
                a = unitAddr(base, 0);
                return;
            }
            cmd = t.col_cmd[ci];
            a = unitAddr(base, base[colLevel] + colsIssued);
            return;
        }
        if (ab && mode == "baseline") {
            if (openStep(cmd, a))
                return;
            cmd = t.col_cmd[ci];
            a = unitAddr(base, base[colLevel] + colsIssued);
            return;
        }
        if (aespa) {
            if (openStep(cmd, a) && cmd == t.cmd_act2)
                return;
            for (int k : actOrder) {
                if (unitCols[k] >= ncols || !unitOpen(k))
                    continue;
                const std::vector<int> x = aespaAddr(k);
                const int sbc = t.col_cmd[int(aespaCls)];
                if (checkCmd(sbc, x, clk)) {
                    cmd = sbc;
                    a = x;
                    return;
                }
            }
            openStep(cmd, a);
            return;
        }
        if (sb && openStep(cmd, a))
            return;
        a = sb ? banks[0] : base;
        a[colLevel] = base[colLevel] + colsIssued;
        cmd = t.col_cmd[ci];
        if (colsIssued + 1 == ncols && t.col_ap_cmd[ci] >= 0 && t.close_rows)
            cmd = t.col_ap_cmd[ci];
    };
    auto legal = [&](int cmd, const std::vector<int> &a, int64_t clk) {
        if (!checkCmd(cmd, a, clk))
            return false;
        if (cmd == t.cmd_act1 || cmd == t.cmd_actab1) {
            const int second =
                cmd == t.cmd_act1 ? t.cmd_act2 : t.cmd_actab2;
            if (!checkCmd(second, a, clk + t.nAAD))
                return false;
        }
        return true;
    };
    auto colClass = [&](int cmd) {
        for (int c = 0; c < int(t.col_cmd.size()); c++) {
            if (t.col_cmd[c] == cmd || t.col_ap_cmd[c] == cmd)
                return c;
        }
        return -1;
    };
    auto setState = [&](int cmd, const std::vector<int> &a) {
        std::vector<Node *> targets;
        if (cmd == t.cmd_actab || cmd == t.cmd_actab1 || cmd == t.cmd_actab2) {
            for (const auto &b : banks)
                targets.push_back(bankNode(b));
        } else if (cmd == t.cmd_act || cmd == t.cmd_act1 ||
                   cmd == t.cmd_act2) {
            targets.push_back(bankNode(a));
        }
        for (Node *n : targets) {
            if (cmd == t.cmd_act1 || cmd == t.cmd_actab1) {
                n->state = Node::State::Activating;
            } else {
                n->state = Node::State::Opened;
                n->row = a[rowLevel];
            }
        }
    };

    int64_t dataEnd = -1;
    bool pushed = false, casPending = false, wck = false;
    int pcmd = -1;
    std::vector<int> paddr;
    const int64_t limit = 1000000;
    for (int64_t clk = 0; clk < limit; clk++) {
        // The plugin's pre_schedule: push the next command if it can go.
        if (!pushed) {
            int cmd = -1;
            std::vector<int> a;
            next(cmd, a, clk);
            if (cmd >= 0 && legal(cmd, a, clk)) {
                pushed = true;
                pcmd = cmd;
                paddr = a;
            }
        }
        // The controller: the pushed command, when it may issue now.
        if (!pushed)
            continue;
        if (casPending) {
            if (!checkCmd(pcmd, paddr, clk))
                continue;
            casPending = false;
        } else {
            if (hbm34 && (first + clk) % 2 != 1 && !isPre[pcmd])
                continue; // HBM4: rising edges (odd clocks)
            if (!checkCmd(pcmd, paddr, clk))
                continue;
            if (lpddr && isAccess[pcmd] && !wck) {
                if (checkCmd(cmdCas, paddr, clk)) {
                    updateCmd(cmdCas, paddr, clk);
                    casPending = true;
                }
                continue;
            }
        }
        // Issue.
        updateCmd(pcmd, paddr, clk);
        setState(pcmd, paddr);
        pushed = false;
        const int c = colClass(pcmd);
        if (c >= 0) {
            colsIssued++;
            dataEnd = std::max(dataEnd, clk + t.done_lat[c]);
            if (isAccess[pcmd])
                wck = true;
            if (aespa) {
                for (int k = 0; k < units; k++) {
                    const std::vector<int> x = aespaAddr(k);
                    if (x == paddr) {
                        unitCols[k]++;
                        break;
                    }
                }
            }
            if (colsIssued == colsTotal)
                return dataEnd;
        } else if (aespa) {
            // PimSequencer::note_aespa_open.
            for (int k = 0; k < units; k++) {
                if (std::find(actOrder.begin(), actOrder.end(), k) ==
                        actOrder.end() &&
                    unitOpen(k)) {
                    actOrder.push_back(k);
                }
            }
        }
    }
    throw std::runtime_error("PimUnloaded: the request did not complete");
}

} // namespace pim
} // namespace gem5
