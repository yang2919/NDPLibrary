#include <algorithm>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"
#include "ramulator/memory_system/i_memory_system.h"
#include "ramulator/memory_system/i_pim_memory_system.h"
#include "ramulator/pim/pim_arb.h"
#include "ramulator/pim/pim_unit.h"

namespace Ramulator {

// gem5's processing-in-memory (PIM) command sequencer: one per channel
// controller. It uses the controller-plugin interface only as the hook every
// controller tick calls and lives outside controller/plugin/. It expands each PIM job into direct
// commands it pushes through the controller's priority buffer (type_id
// kPimTypeId, no preq), one at a time and only when the command can issue,
// arbitrates against host requests, and computes each job's completion.
class PimSequencer final : public IControllerPlugin, public IPimSequencer, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, PimSequencer, "PimSequencer")

 private:
  enum class AbMode { Ideal, AiM, Baseline, Aespa };

  struct Job {
    PimJob job;
    std::function<void(uint64_t)> on_last;
    std::function<void(uint64_t, Clk_t)> on_done;
    // The banks the job's rows must be open in: one (SB), a pair (SB pair),
    // every bank of the unit (AB), at the job's row.
    std::vector<AddrVec_t> banks;
    int cols_issued = 0;  // column commands issued
    int cols_total = 0;   // ncols, or ncols per bank (pair) under AESPA
    bool started = false;
    bool column_issued = false;  // a column of the job has issued
    bool needed_act = false;
    std::vector<int> opened;  // flat banks the job activated (re-opens count against it)
    bool ab_opened = false;   // the job issued an all-bank ACT
    bool closing = false;     // the columns are done; closing PREs remain
    Clk_t data_end = -1;      // latest data end of its column commands
    // AESPA: banks (or pairs) in activation order and
    // their next column; units = indexes into banks (pairs: the even bank).
    std::vector<int> act_order;
    std::vector<int> unit_cols;
  };
  struct Pending {
    uint64_t id;
    Clk_t done;
    std::function<void(uint64_t, Clk_t)> on_done;
  };
  struct Pushed {
    bool valid = false;
    int command = -1;
    AddrVec_t addr;
    int unit = -1;
  };

  // Parameters
  bool m_close_rows = false;
  std::string m_ab_mode;
  std::string m_arb_name;
  bool m_check_state = true;
  bool m_debug = false;
  std::vector<std::string> m_test_jobs;
  std::vector<int> m_test_write_drain;

  ControllerBase* m_ctrl = nullptr;
  const DRAMSpec* m_spec = nullptr;
  IPimMemorySystem* m_pimsys = nullptr;
  std::unique_ptr<PimArbPolicy> m_arb;

  int m_unit_level = 0, m_bank_level = 0, m_row_level = 0, m_col_level = 0;
  int m_units = 1;
  int m_state_opened = -1, m_state_closed = -1, m_state_activating = -1;
  int m_cmd_act = -1, m_cmd_act1 = -1, m_cmd_act2 = -1, m_cmd_prepb = -1, m_cmd_preab = -1;
  int m_cmd_actab = -1, m_cmd_actab1 = -1, m_cmd_actab2 = -1;
  AbMode m_mode = AbMode::Baseline;
  std::vector<int> m_col_cmd, m_col_ap_cmd;  // per PimClass
  std::vector<int> m_done_lat;               // per PimClass, in controller clocks
  int m_nAAD = -1;

  std::vector<std::deque<Job>> m_jobs;  // per unit, in order
  std::vector<bool> m_gated;            // per unit: host work stopped
  std::vector<Pending> m_pending;       // jobs past their last command, waiting for Done
  Pushed m_pushed;
  std::vector<Clk_t> m_act1_clk;                     // per flat bank: our ACTsb1, waiting for ACTsb2
  std::vector<Clk_t> m_actab1_clk;                   // per unit: our ACTab1, waiting for ACTab2
  std::vector<std::vector<AddrVec_t>> m_unit_banks;  // per unit: every bank, flat order
  int m_external_backlog = 0;
  bool m_drain_on = false;
  int m_next_unit = 0;
  PimSide m_last_side = PimSide::Host;
  bool m_last_was_pre = false;

  // Test mode (jobs without a memory system)
  struct TestJob {
    Clk_t at;
    PimJob job;
    bool submitted = false;
  };
  std::vector<TestJob> m_tests;
  std::vector<size_t> s_test_last, s_test_done;

  // Statistics: every name starts with pim_
  size_t s_jobs = 0, s_acts = 0, s_pres = 0, s_ab_acts = 0, s_ab_pres = 0, s_ap_cols = 0;
  size_t s_row_hits = 0, s_reopens = 0, s_interrupts = 0, s_gated_cycles = 0, s_drain_cycles = 0;
  size_t s_sync_cmds = 0;  // controller-inserted CAS/RCKSTRT before a PIM command
  std::vector<size_t> s_col_cmds;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_close_rows, bool, "close_rows").default_val(false);
    RAMULATOR_PARSE_PARAM(m_ab_mode, std::string, "ab_mode").default_val("baseline");
    RAMULATOR_PARSE_PARAM(m_arb_name, std::string, "arb_policy").default_val("frfcfs");
    RAMULATOR_PARSE_PARAM(m_check_state, bool, "check_state").default_val(true);
    RAMULATOR_PARSE_PARAM(m_debug, bool, "debug").default_val(false);
    // Test only: "<clk>:<class>:<a0>/<a1>/...:<ncols>" jobs submitted at clk.
    RAMULATOR_PARSE_PARAM(m_test_jobs, std::vector<std::string>, "test_jobs").default_val(std::vector<std::string>{});
    // Test only: [on, off] controller clocks of a write-drain request.
    RAMULATOR_PARSE_PARAM(m_test_write_drain, std::vector<int>, "test_write_drain").default_val(std::vector<int>{});
    m_arb = make_pim_arb_policy(m_arb_name);
    if (m_ab_mode == "ideal") {
      m_mode = AbMode::Ideal;
    } else if (m_ab_mode == "aim") {
      m_mode = AbMode::AiM;
    } else if (m_ab_mode == "baseline") {
      m_mode = AbMode::Baseline;
    } else if (m_ab_mode == "aespa") {
      m_mode = AbMode::Aespa;
    } else {
      throw std::runtime_error("PimSequencer: unknown ab_mode '" + m_ab_mode +
                               "' (expected ideal, aim, baseline or aespa)");
    }
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    m_ctrl = cast_parent<ControllerBase>();
    m_spec = m_ctrl->m_device.m_spec;
    const auto& s = *m_spec;
    m_unit_level = pim_unit_level(s);
    m_bank_level = s.get_level_id("Bank");
    m_row_level = s.get_level_id("Row");
    m_col_level = s.get_level_id("Column");
    m_units = m_unit_level > 0 ? s.organization.level_sizes[m_unit_level] : 1;
    m_state_opened = s.get_state_id("Opened");
    m_state_closed = s.get_state_id("Closed");
    m_state_activating = s.has_state("Activating") ? s.get_state_id("Activating") : -1;

    if (!s.has_command("PIM_SB_RD")) {
      throw std::runtime_error("PimSequencer: the DRAM standard " + s.standard_name +
                               " has no PIM commands (use a *_PIM standard)");
    }
    if (s.has_command("ACTsb1")) {
      m_cmd_act1 = s.get_command_id("ACTsb1");
      m_cmd_act2 = s.get_command_id("ACTsb2");
      m_nAAD = s.get_timing_value("nAAD");
    } else {
      m_cmd_act = s.get_command_id("ACT");
    }
    m_cmd_prepb = s.get_command_id("PREpb");
    m_cmd_preab = s.get_command_id("PREab");
    if (s.has_command("ACTab1")) {
      m_cmd_actab1 = s.get_command_id("ACTab1");
      m_cmd_actab2 = s.get_command_id("ACTab2");
    } else {
      m_cmd_actab = s.get_command_id("ACTab");
    }

    static const char* const col[] = {"PIM_SB_RD",  "PIM_SB_RD2",   "PIM_SB_WR",    "PIM_AB_EW",
                                      "PIM_AB_MAC", "PIM_AB_BKREG", "PIM_AB_REGBK", "PIM_GB_RD",
                                      "PIM_GB_WR",  "PIM_REG_RD",   "PIM_REG_WR"};
    static const char* const ap[] = {"PIM_SB_RDA", "PIM_SB_RD2A", "PIM_SB_WRA"};
    const int n = static_cast<int>(PimClass::COUNT);
    m_col_cmd.resize(n);
    m_col_ap_cmd.assign(n, -1);
    for (int c = 0; c < n; c++) {
      m_col_cmd[c] = s.get_command_id(col[c]);
    }
    for (int c = 0; c < 3; c++) {
      m_col_ap_cmd[c] = s.get_command_id(ap[c]);
    }
    // Done after the last column (nEX_AB_RD or nEX_AB_WR for AB). A
    // non-bank load completes after the host read path, nCL_x + (read_latency
    // - host CAS latency): nCL_x + burst, and on GDDR7 one more clock.
    int nbl = s.has_timing("nBL") ? s.get_timing_value("nBL") : s.get_timing_value("nBL_min");
    auto t = [&](const char* name) { return s.get_timing_value(name); };
    int cas = s.has_timing("nCL") ? t("nCL") : t("nRL");
    int rd_path = static_cast<int>(s.read_latency) - cas;
    m_done_lat = {t("nSB_RD_DONE"),   t("nSB_RD_DONE"),       t("nSB_WR_DONE"),   t("nEX_AB_RD"),
                  t("nEX_AB_RD"),     t("nEX_AB_RD"),         t("nEX_AB_WR"),     t("nCL_GB") + rd_path,
                  t("nCWL_GB") + nbl, t("nCL_REG") + rd_path, t("nCWL_REG") + nbl};

    m_jobs.assign(m_units, {});
    m_gated.assign(m_units, false);
    m_act1_clk.assign(m_ctrl->m_device.m_bank_nodes.size(), -1);
    m_actab1_clk.assign(m_units, -1);
    m_unit_banks.assign(m_units, {});
    for (auto* b : m_ctrl->m_device.m_bank_nodes) {
      AddrVec_t a(s.level_count, 0);
      for (auto* n = b; n != nullptr; n = n->m_parent_node) {
        a[n->m_level] = n->m_node_id;
      }
      m_unit_banks[unit_of(a)].push_back(a);
    }
    m_ctrl->set_host_filter([this](const Request& req) { return !m_gated[unit_of(req.addr_vec)]; });

    m_pimsys = dynamic_cast<IPimMemorySystem*>(memory_system);
    if (m_pimsys) {
      m_pimsys->register_sequencer(m_ctrl->m_channel_id, this);
    }

    s_col_cmds.assign(n, 0);
    m_stats.add("pim_jobs", s_jobs);
    for (int c = 0; c < n; c++) {
      m_stats.add(fmt::format("pim_col_cmds_{}", pim_class_name(static_cast<PimClass>(c))), s_col_cmds[c]);
    }
    m_stats.add("pim_acts", s_acts);
    m_stats.add("pim_pres", s_pres);
    m_stats.add("pim_ab_acts", s_ab_acts);
    m_stats.add("pim_ab_pres", s_ab_pres);
    m_stats.add("pim_ap_cols", s_ap_cols);
    m_stats.add("pim_row_hits", s_row_hits);
    m_stats.add("pim_reopens", s_reopens);
    m_stats.add("pim_interrupts", s_interrupts);
    m_stats.add("pim_gated_cycles", s_gated_cycles);
    m_stats.add("pim_drain_cycles", s_drain_cycles);
    m_stats.add("pim_sync_cmds", s_sync_cmds);

    parse_test_jobs();
  }

  // ── IPimSequencer ─────────────────────────────────────────────────────

  void submit(const PimJob& job, std::function<void(uint64_t)> on_last_command,
              std::function<void(uint64_t, Clk_t)> on_done) override {
    if (static_cast<int>(job.addr.size()) != m_spec->level_count || job.ncols < 1) {
      throw std::runtime_error(fmt::format("PimSequencer: job {} has a malformed address or column count", job.id));
    }
    Job j;
    j.job = job;
    j.on_last = std::move(on_last_command);
    j.on_done = std::move(on_done);
    j.cols_total = job.ncols;
    if (pim_class_is_sb(job.cls)) {
      AddrVec_t a = job.addr;
      if (job.cls == PimClass::SbRd2) {
        a = m_ctrl->m_device.pair_addr_vec(job.addr, 0);
        j.banks = {a, m_ctrl->m_device.pair_addr_vec(job.addr, 1)};
      } else {
        j.banks = {a};
      }
    } else if (pim_class_is_ab(job.cls)) {
      for (AddrVec_t b : m_unit_banks[unit_of(job.addr)]) {
        b[m_row_level] = job.addr[m_row_level];
        b[m_col_level] = job.addr[m_col_level];
        j.banks.push_back(b);
      }
      if (m_mode == AbMode::Aespa) {
        // AB operations become SB commands per bank, or per pair for AB EW.
        int units = static_cast<int>(j.banks.size());
        if (job.cls == PimClass::AbEw) {
          units /= 2;
        }
        j.unit_cols.assign(units, 0);
        j.cols_total = job.ncols * units;
      }
    }
    m_jobs[unit_of(job.addr)].push_back(std::move(j));
    s_jobs++;
  }

  void set_external_backlog(int packets) override {
    m_external_backlog = packets;
  }

  void note_write_drain(bool on) override {
    m_drain_on = on;
  }

  void describe(PimTimingInfo& info) const override {
    info.col_cmd = m_col_cmd;
    info.col_ap_cmd = m_col_ap_cmd;
    info.done_lat = m_done_lat;
    info.cmd_act = m_cmd_act;
    info.cmd_act1 = m_cmd_act1;
    info.cmd_act2 = m_cmd_act2;
    info.cmd_actab = m_cmd_actab;
    info.cmd_actab1 = m_cmd_actab1;
    info.cmd_actab2 = m_cmd_actab2;
    info.nAAD = m_nAAD;
    static const char* const modes[] = {"ideal", "aim", "baseline", "aespa"};
    info.ab_mode = modes[static_cast<int>(m_mode)];
    info.close_rows = m_close_rows;
  }

  bool idle() const override {
    if (m_pushed.valid || !m_pending.empty()) {
      return false;
    }
    for (const auto& q : m_jobs) {
      if (!q.empty()) {
        return false;
      }
    }
    return true;
  }

  // ── Controller hooks ──────────────────────────────────────────────────

  void pre_schedule() override {
    Clk_t clk = m_ctrl->m_clk;
    run_test_inputs(clk);
    for (bool g : m_gated) {
      if (g) {
        s_gated_cycles++;
        break;
      }
    }
    if (m_drain_on) {
      s_drain_cycles++;
    }
    if (m_pushed.valid) {
      return;  // one PIM command at a time
    }
    // The ACT-2 of a PIM ACT-1 that has issued goes first, whatever the
    // arbitration and before refresh: it must follow within nAAD (LPDDR6),
    // as a host ACT-2 does, and the controller holds back closing commands
    // to its bank until then (README.gem5, change 18).
    if (push_pending_act2()) {
      return;
    }
    // Push rule 1: refresh and row-policy entries go first.
    if (m_ctrl->priority_buffer_size() != 0) {
      return;
    }
    bool peeked = false;
    ControllerBase::HostPeek host;
    for (int k = 0; k < m_units; k++) {
      int u = (m_next_unit + k) % m_units;
      if (m_jobs[u].empty()) {
        if (m_arb->gate() == PimGate::WhileSelected) {
          m_gated[u] = false;  // no PIM work: the host side
        }
        continue;
      }
      Job& j = m_jobs[u].front();
      if (j.closing && !next_close(j, u)) {
        // Its banks closed before its closing PREs (a refresh's PREab):
        // nothing is left to close, so the job is done.
        finish_head(u);
        continue;
      }
      if (m_mode == AbMode::Aespa && pim_class_is_ab(j.job.cls) && !j.closing) {
        // Banks that already hold the job's row (kept open for it) join the
        // activation order: no ACT of the job will.
        note_aespa_open(j);
      }
      int cmd = -1;
      AddrVec_t addr;
      bool ready = next_command(j, u, cmd, addr);
      PimArbView v;
      v.pim_pending = true;
      v.pim_started = j.started;
      v.pim_ready = ready;
      v.pim_row_hit = ready && is_column(cmd);
      v.host_queued = m_ctrl->read_buffer_size() + m_ctrl->write_buffer_size() > 0 || m_external_backlog > 0;
      v.last_side = m_last_side;
      v.last_was_pre = m_last_was_pre;
      if (!j.started && ready && m_arb->gate() == PimGate::WhileStarted) {
        if (!peeked) {
          host = m_ctrl->peek_host();
          peeked = true;
        }
        v.host_ready = host.valid;
        v.host_row_hit = host.row_hit;
        v.host_older = host.valid && host.arrive < j.job.accept_clk;
      }
      PimSide side = m_arb->select(v);
      if (m_arb->gate() == PimGate::WhileSelected) {
        m_gated[u] = side == PimSide::Pim;  // AsyncDIMM: the PIM side holds the unit
      }
      if (!ready || side != PimSide::Pim) {
        continue;
      }
      if (!j.started) {
        j.started = true;
        if (m_arb->gate() == PimGate::WhileStarted) {
          m_gated[u] = true;
        }
      }
      push(cmd, addr, u);
      m_next_unit = (u + 1) % m_units;
      return;
    }
  }

  void on_issue(const Request& req) override {
    const auto& s = *m_spec;
    if (req.type_id != kPimTypeId) {
      // Only host requests (types 0 and 1) are AsyncDIMM's host side: refresh,
      // row-policy and other controller-made commands (type -1) leave the side
      // as it was.
      if (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write) {
        m_last_side = PimSide::Host;
        m_last_was_pre = s.command_meta[req.command].is_closing;
      }
      bool column = req.command == req.final_command &&
                    (req.type_id == Request::Type::Read || req.type_id == Request::Type::Write);
      if (column && m_pimsys) {
        m_pimsys->host_column_issued(m_ctrl->m_channel_id, req.addr, req.type_id == Request::Type::Read);
      }
      if (req.type_id >= 0 && !req.addr_vec.empty()) {
        int u = unit_of(req.addr_vec);
        if (!m_jobs[u].empty() && m_jobs[u].front().started) {
          s_interrupts++;  // host work between the commands of a started job
        }
      }
      return;
    }
    if (m_pushed.valid && req.command != req.final_command && req.final_command == m_pushed.command) {
      // A command the controller issues before ours on its behalf: the
      // LPDDR6 WCK sync CAS (README.gem5, change 8). GDDR7's RCKSTRT is a
      // separate controller request of type -1, not counted here.
      s_sync_cmds++;
      return;
    }
    if (!m_pushed.valid || req.command != m_pushed.command || req.addr_vec != m_pushed.addr) {
      throw std::runtime_error(
          fmt::format("PimSequencer: issued PIM command {} is not the pushed one", s.command_names[req.command]));
    }
    m_last_side = PimSide::Pim;
    m_last_was_pre = s.command_meta[req.command].is_closing;
    int u = m_pushed.unit;
    m_pushed.valid = false;
    Job& j = m_jobs[u].front();
    Clk_t clk = m_ctrl->m_clk;
    if (m_check_state) {
      check_after_issue(req.command, req.addr_vec, clk);
    }
    if (req.command == m_cmd_act || req.command == m_cmd_act2 || req.command == m_cmd_actab ||
        req.command == m_cmd_actab2) {
      bool ab = req.command == m_cmd_actab || req.command == m_cmd_actab2;
      (ab ? s_ab_acts : s_acts)++;
      j.needed_act = true;
      // A re-open: the job opens again a row it had opened, which another
      // command (refresh, a host row miss, the row policy) closed.
      if (ab) {
        s_reopens += j.ab_opened ? 1 : 0;
        j.ab_opened = true;
      } else {
        int id = m_ctrl->m_device.get_flat_bank_id(req.addr_vec);
        if (std::find(j.opened.begin(), j.opened.end(), id) != j.opened.end()) {
          s_reopens++;
        } else {
          j.opened.push_back(id);
        }
      }
      if (m_mode == AbMode::Aespa && pim_class_is_ab(j.job.cls)) {
        note_aespa_open(j);
      }
      if (req.command == m_cmd_actab2) {
        m_actab1_clk[u] = -1;
      }
      if (req.command == m_cmd_act2) {
        m_act1_clk[m_ctrl->m_device.get_flat_bank_id(req.addr_vec)] = -1;
      }
    } else if (req.command == m_cmd_prepb) {
      s_pres++;
    } else if (req.command == m_cmd_preab) {
      s_ab_pres++;
    } else if (req.command == m_cmd_act1) {
      m_act1_clk[m_ctrl->m_device.get_flat_bank_id(req.addr_vec)] = clk;
    } else if (req.command == m_cmd_actab1) {
      m_actab1_clk[u] = clk;
    }
    if (j.closing) {
      if (!next_close(j, u)) {
        finish_head(u);
      }
      return;
    }
    if (!is_column(req.command)) {
      return;
    }
    int c = column_class(req.command);
    s_col_cmds[static_cast<int>(j.job.cls)]++;
    if (req.command == m_col_ap_cmd[c]) {
      s_ap_cols++;
    }
    if (!j.column_issued && !pim_class_is_non_bank(j.job.cls) && !j.needed_act) {
      s_row_hits++;
    }
    j.column_issued = true;
    j.cols_issued++;
    j.data_end = std::max(j.data_end, clk + m_done_lat[c]);
    if (!j.unit_cols.empty()) {
      j.unit_cols[aespa_unit_of(j, req.addr_vec)]++;
    }
    if (j.cols_issued == j.cols_total) {
      if (pim_class_is_ab(j.job.cls) && close_after(j, u)) {
        j.closing = true;
        if (next_close(j, u)) {
          return;  // closing PREs follow
        }
      }
      finish_head(u);
    }
  }

  void post_schedule() override {
    Clk_t clk = m_ctrl->m_clk;
    for (size_t i = 0; i < m_pending.size();) {
      if (m_pending[i].done <= clk) {
        Pending p = std::move(m_pending[i]);
        m_pending.erase(m_pending.begin() + i);
        if (p.on_done) {
          p.on_done(p.id, p.done);
        }
      } else {
        i++;
      }
    }
  }

 private:
  int unit_of(const AddrVec_t& addr) const {
    if (m_unit_level == 0) {
      return 0;
    }
    int u = addr[m_unit_level];
    return u < 0 ? 0 : u;
  }

  bool is_column(int cmd) const {
    for (size_t c = 0; c < m_col_cmd.size(); c++) {
      if (cmd == m_col_cmd[c] || cmd == m_col_ap_cmd[c]) {
        return true;
      }
    }
    return false;
  }

  DRAMNode* bank_node(const AddrVec_t& a) const {
    return m_ctrl->m_device.m_bank_nodes[m_ctrl->m_device.get_flat_bank_id(a)];
  }

  bool row_open(const AddrVec_t& a) const {
    auto* b = bank_node(a);
    return b->m_state == m_state_opened && b->m_row_state.count(a[m_row_level]);
  }

  // Push rule 2: no host work in flight in the unit, none in the active
  // buffer and no bank activating, apart from the job's own ACTsb1.
  bool unit_quiet(int u) const {
    AddrVec_t scope(m_spec->level_count, -1);
    if (m_unit_level > 0) {
      scope[m_unit_level] = u;
    }
    if (m_ctrl->has_active_in(scope)) {
      return false;
    }
    if (m_state_activating < 0 || m_actab1_clk[u] >= 0) {
      return true;
    }
    const auto& banks = m_ctrl->m_device.m_bank_nodes;
    for (size_t i = 0; i < banks.size(); i++) {
      if (banks[i]->m_state == m_state_activating && DRAMDevice::bank_matches(banks[i], scope) && m_act1_clk[i] < 0) {
        return false;
      }
    }
    return true;
  }

  // The job's next command and address; true when it can be pushed now
  // (push rules 2 and 3). The choice follows the banks' state, so rows that
  // another command closed are opened again.
  bool next_command(const Job& j, int u, int& cmd, AddrVec_t& addr) const {
    if (j.closing) {
      return close_command(j, u, cmd, addr) && legal(cmd, addr, u);
    }
    if (pim_class_is_ab(j.job.cls)) {
      return ab_next(j, u, cmd, addr) && legal(cmd, addr, u);
    }
    if (pim_class_is_sb(j.job.cls) && open_step(j.banks, cmd, addr)) {
      return legal(cmd, addr, u);
    }
    int c = static_cast<int>(j.job.cls);
    addr = pim_class_is_sb(j.job.cls) ? j.banks[0] : j.job.addr;
    addr[m_col_level] = j.job.addr[m_col_level] + j.cols_issued;
    cmd = m_col_cmd[c];
    if (j.cols_issued + 1 == j.job.ncols && m_col_ap_cmd[c] >= 0 && close_after(j, u)) {
      cmd = m_col_ap_cmd[c];
    }
    return legal(cmd, addr, u);
  }

  // The next per-bank row command for banks, in order: PREpb when another
  // row is open, ACTsb2 after our ACTsb1, else ACT; false when all hold their row.
  bool open_step(const std::vector<AddrVec_t>& banks, int& cmd, AddrVec_t& addr) const {
    for (const auto& b : banks) {
      if (row_open(b)) {
        continue;
      }
      auto* node = bank_node(b);
      if (node->m_state == m_state_opened) {
        cmd = m_cmd_prepb;  // another row is open
      } else if (m_state_activating >= 0 && node->m_state == m_state_activating) {
        cmd = m_cmd_act2;  // our ACTsb1 issued
      } else {
        cmd = m_cmd_act >= 0 ? m_cmd_act : m_cmd_act1;
      }
      addr = b;
      return true;
    }
    return false;
  }

  AddrVec_t unit_addr(const Job& j, int col) const {
    AddrVec_t a = j.job.addr;
    for (int l = m_unit_level + 1; l < m_row_level; l++) {
      a[l] = -1;
    }
    a[m_col_level] = col;
    return a;
  }

  // All-bank jobs: the opening, columns and re-opening of each mode
  // (ideal, aim, baseline, aespa).
  bool ab_next(const Job& j, int u, int& cmd, AddrVec_t& addr) const {
    int c = static_cast<int>(j.job.cls);
    if (m_mode == AbMode::Ideal || m_mode == AbMode::AiM) {
      bool all = true, any_open = false, all_activating = true;
      for (const auto& b : j.banks) {
        auto* n = bank_node(b);
        all = all && row_open(b);
        any_open = any_open || n->m_state == m_state_opened;
        all_activating = all_activating && m_state_activating >= 0 && n->m_state == m_state_activating;
      }
      if (!all) {
        if (any_open) {
          // Close the open banks first: Ideal with PREab, AiM bank by bank.
          if (m_mode == AbMode::Ideal) {
            cmd = m_cmd_preab;
            addr = unit_addr(j, 0);
            return true;
          }
          for (const auto& b : j.banks) {
            if (bank_node(b)->m_state == m_state_opened) {
              cmd = m_cmd_prepb;
              addr = b;
              return true;
            }
          }
        }
        cmd = (m_cmd_actab >= 0) ? m_cmd_actab : (all_activating ? m_cmd_actab2 : m_cmd_actab1);
        addr = unit_addr(j, 0);
        return true;
      }
      cmd = m_col_cmd[c];
      addr = unit_addr(j, j.job.addr[m_col_level] + j.cols_issued);
      return true;
    }
    if (m_mode == AbMode::Baseline) {
      if (open_step(j.banks, cmd, addr)) {
        return true;
      }
      cmd = m_col_cmd[c];
      addr = unit_addr(j, j.job.addr[m_col_level] + j.cols_issued);
      return true;
    }
    // AESPA: per cycle the earliest legal of the next column of an open bank
    // (in activation order) and the next per-bank row command. A pending
    // ACTsb2 goes first: it must follow its ACTsb1 within nAAD (LPDDR6).
    if (open_step(j.banks, cmd, addr) && cmd == m_cmd_act2) {
      return true;
    }
    for (int k : j.act_order) {
      if (j.unit_cols[k] >= j.job.ncols) {
        continue;
      }
      AddrVec_t a = aespa_addr(j, k);
      bool open = row_open(a) && (j.job.cls != PimClass::AbEw || row_open(m_ctrl->m_device.pair_addr_vec(a, 1)));
      if (!open) {
        continue;  // lost its row: the row command below reopens it
      }
      int sb = m_col_cmd[static_cast<int>(aespa_class(j.job.cls))];
      if (m_ctrl->check_timing(sb, a)) {
        cmd = sb;
        addr = a;
        return true;
      }
    }
    return open_step(j.banks, cmd, addr);
  }

  static PimClass aespa_class(PimClass c) {
    if (c == PimClass::AbEw) {
      return PimClass::SbRd2;
    }
    return c == PimClass::AbRegbk ? PimClass::SbWr : PimClass::SbRd;
  }

  // AESPA unit k: bank k, or the pair whose even bank is 2k-th in the unit.
  AddrVec_t aespa_addr(const Job& j, int k) const {
    AddrVec_t a = j.banks[j.job.cls == PimClass::AbEw ? 2 * k : k];
    a[m_col_level] = j.job.addr[m_col_level] + j.unit_cols[k];
    return a;
  }

  int aespa_unit_of(const Job& j, const AddrVec_t& addr) const {
    int id = m_ctrl->m_device.get_flat_bank_id(addr);
    for (size_t k = 0; k < j.unit_cols.size(); k++) {
      if (m_ctrl->m_device.get_flat_bank_id(aespa_addr(j, static_cast<int>(k))) == id) {
        return static_cast<int>(k);
      }
    }
    throw std::runtime_error("PimSequencer: AESPA column to a bank outside its job");
  }

  // AESPA: a bank (pair) whose row is now open joins the activation order.
  void note_aespa_open(Job& j) {
    for (size_t k = 0; k < j.unit_cols.size(); k++) {
      int ki = static_cast<int>(k);
      if (std::find(j.act_order.begin(), j.act_order.end(), ki) != j.act_order.end()) {
        continue;
      }
      AddrVec_t a = aespa_addr(j, ki);
      bool open = row_open(a) && (j.job.cls != PimClass::AbEw || row_open(m_ctrl->m_device.pair_addr_vec(a, 1)));
      if (open) {
        j.act_order.push_back(ki);
      }
    }
  }

  // Closing an AB job's rows: Ideal one PREab; the others
  // PREpb per open bank (AESPA in activation order).
  bool close_command(const Job& j, int u, int& cmd, AddrVec_t& addr) const {
    if (m_mode == AbMode::Ideal) {
      cmd = m_cmd_preab;
      addr = unit_addr(j, 0);
      return true;
    }
    std::vector<int> order;
    if (m_mode == AbMode::Aespa) {
      for (int k : j.act_order) {
        order.push_back(j.job.cls == PimClass::AbEw ? 2 * k : k);
        if (j.job.cls == PimClass::AbEw) {
          order.push_back(2 * k + 1);
        }
      }
    } else {
      for (size_t i = 0; i < j.banks.size(); i++) {
        order.push_back(static_cast<int>(i));
      }
    }
    for (int i : order) {
      if (bank_node(j.banks[i])->m_state == m_state_opened) {
        cmd = m_cmd_prepb;
        addr = j.banks[i];
        return true;
      }
    }
    return false;
  }

  // After a closing PRE: whether another remains.
  bool next_close(const Job& j, int u) const {
    if (m_mode == AbMode::Ideal) {
      for (const auto& b : j.banks) {
        if (bank_node(b)->m_state == m_state_opened) {
          return true;
        }
      }
      return false;
    }
    int cmd;
    AddrVec_t addr;
    return close_command(j, u, cmd, addr);
  }

  // With close_rows (a closing row policy), the last SB column closes its
  // rows unless the unit's next job needs the same rows.
  bool close_after(const Job& j, int u) const {
    if (!m_close_rows) {
      return false;
    }
    if (m_jobs[u].size() > 1) {
      const Job& n = m_jobs[u][1];
      if (pim_class_is_ab(j.job.cls) && pim_class_is_ab(n.job.cls)) {
        return n.job.addr[m_row_level] != j.job.addr[m_row_level];
      }
      if (pim_class_is_sb(n.job.cls) && n.banks.size() == j.banks.size()) {
        bool same = true;
        for (size_t i = 0; i < j.banks.size(); i++) {
          AddrVec_t a = j.banks[i], b = n.banks[i];
          a[m_col_level] = b[m_col_level] = 0;
          same = same && a == b;
        }
        if (same) {
          return false;
        }
      }
    }
    return true;
  }

  bool legal(int cmd, const AddrVec_t& addr, int u) const {
    if (!m_jobs[u].front().started && !unit_quiet(u)) {
      return false;
    }
    if (m_jobs[u].front().started && m_ctrl->has_active_in(unit_scope(u))) {
      return false;
    }
    if (!m_ctrl->check_timing(cmd, addr)) {
      return false;
    }
    // LPDDR6: a first ACT phase only when its second phase will be legal
    // within nAAD; otherwise the second phase would miss its deadline.
    if (cmd == m_cmd_act1 || cmd == m_cmd_actab1) {
      int second = cmd == m_cmd_act1 ? m_cmd_act2 : m_cmd_actab2;
      if (!m_ctrl->m_device.check_timing(second, addr, m_ctrl->m_clk + m_nAAD)) {
        return false;
      }
    }
    return !m_check_state || state_allows(cmd, addr);
  }

  AddrVec_t unit_scope(int u) const {
    AddrVec_t scope(m_spec->level_count, -1);
    if (m_unit_level > 0) {
      scope[m_unit_level] = u;
    }
    return scope;
  }

  // The state checks before a push.
  int column_class(int cmd) const {
    for (size_t k = 0; k < m_col_cmd.size(); k++) {
      if (cmd == m_col_cmd[k] || cmd == m_col_ap_cmd[k]) {
        return static_cast<int>(k);
      }
    }
    return -1;
  }

  bool state_allows(int cmd, const AddrVec_t& addr) const {
    if (is_column(cmd)) {
      int c = column_class(cmd);
      if (pim_class_is_non_bank(static_cast<PimClass>(c))) {
        return true;
      }
      bool ok = true;
      m_ctrl->m_device.for_each_target_bank(cmd, addr, [&](int id) {
        auto* b = m_ctrl->m_device.m_bank_nodes[id];
        ok = ok && b->m_state == m_state_opened && b->m_row_state.count(addr[m_row_level]);
      });
      return ok;
    }
    if (cmd == m_cmd_act || cmd == m_cmd_act1) {
      return bank_node(addr)->m_state == m_state_closed;
    }
    if (cmd == m_cmd_actab || cmd == m_cmd_actab1) {
      // An all-bank ACT only when every bank of the unit is closed.
      bool closed = true;
      m_ctrl->m_device.for_each_target_bank(
          cmd, addr, [&](int id) { closed = closed && m_ctrl->m_device.m_bank_nodes[id]->m_state == m_state_closed; });
      return closed;
    }
    if (cmd == m_cmd_actab2) {
      Clk_t act1 = m_actab1_clk[unit_of(addr)];
      return act1 >= 0 && m_ctrl->m_clk - act1 <= m_nAAD;
    }
    if (cmd == m_cmd_act2) {
      Clk_t act1 = m_act1_clk[m_ctrl->m_device.get_flat_bank_id(addr)];
      return act1 >= 0 && m_ctrl->m_clk - act1 <= m_nAAD;  // ACTsb2 within nAAD of ACTsb1
    }
    return true;
  }

  void check_after_issue(int cmd, const AddrVec_t& addr, Clk_t clk) {
    const auto& s = *m_spec;
    if (cmd == m_cmd_actab2) {
      Clk_t act1 = m_actab1_clk[unit_of(addr)];
      if (act1 < 0 || clk - act1 > m_nAAD) {
        throw std::runtime_error(fmt::format("PimSequencer: ACTab2 at clk {} not within nAAD of ACTab1", clk));
      }
    }
    if (cmd == m_cmd_act2) {
      int id = m_ctrl->m_device.get_flat_bank_id(addr);
      if (m_act1_clk[id] < 0 || clk - m_act1_clk[id] > m_nAAD) {
        throw std::runtime_error(fmt::format("PimSequencer: ACTsb2 at clk {} not within nAAD of ACTsb1", clk));
      }
      m_act1_clk[id] = -1;
    }
    if (is_column(cmd)) {
      // A closing column has already closed its banks; check the open state
      // only for the others.
      if (!s.command_meta[cmd].is_closing && !state_allows(cmd, addr)) {
        throw std::runtime_error(
            fmt::format("PimSequencer: {} at clk {} issued to a bank without its row", s.command_names[cmd], clk));
      }
    }
  }

  void push(int cmd, const AddrVec_t& addr, int u) {
    Request req(addr, Request::Cmd, cmd);
    req.type_id = kPimTypeId;
    if (!m_ctrl->priority_send(req)) {
      return;  // the buffer refused; try again next cycle
    }
    m_pushed.valid = true;
    m_pushed.command = cmd;
    m_pushed.addr = addr;
    m_pushed.unit = u;
    if (m_debug) {
      std::ostringstream os;
      for (int a : addr) {
        os << a << " ";
      }
      fmt::print("PimSequencer ch{} clk {}: push {} [{}]\n", m_ctrl->m_channel_id, m_ctrl->m_clk,
                 m_spec->command_names[cmd], os.str());
    }
  }

  // The head job of unit u issued its last command: Done at its latest data
  // end; the next job of the unit may start.
  // Push the ACT-2 (ACTsb2 or ACTab2) a started job needs next; true if
  // one was pushed.
  bool push_pending_act2() {
    for (int u = 0; u < m_units; u++) {
      if (m_jobs[u].empty() || !m_jobs[u].front().started) {
        continue;
      }
      // The bank (or unit) between the job's ACT-1 and its ACT-2, by the
      // banks' state: other banks of the job may be closed meanwhile. Only
      // its timing gates it, as for a host ACT-2: a host request active in
      // the unit may be waiting for this bank.
      const Job& j = m_jobs[u].front();
      int cmd = -1;
      AddrVec_t addr;
      if (m_cmd_actab2 >= 0 && m_actab1_clk[u] >= 0) {
        cmd = m_cmd_actab2;
        addr = unit_addr(j, 0);
      } else if (m_cmd_act2 >= 0 && m_state_activating >= 0) {
        for (const auto& b : j.banks) {
          // Ours: a host ACT-1 also leaves a bank activating.
          if (bank_node(b)->m_state == m_state_activating && m_act1_clk[m_ctrl->m_device.get_flat_bank_id(b)] >= 0) {
            cmd = m_cmd_act2;
            addr = b;
            break;
          }
        }
      }
      if (cmd >= 0 && m_ctrl->check_timing(cmd, addr)) {
        push(cmd, addr, u);
        return true;
      }
    }
    return false;
  }

  void finish_head(int u) {
    Job j = std::move(m_jobs[u].front());
    m_jobs[u].pop_front();
    if (m_arb->gate() != PimGate::WhileSelected) {
      m_gated[u] = false;
    }
    if (j.on_last) {
      j.on_last(j.job.id);
    }
    m_pending.push_back({j.job.id, j.data_end, std::move(j.on_done)});
  }

  // ── Test mode ─────────────────────────────────────────────────────────

  void parse_test_jobs() {
    static const char* const names[] = {"SbRd",    "SbRd2", "SbWr", "AbEw",  "AbMac", "AbBkreg",
                                        "AbRegbk", "GbRd",  "GbWr", "RegRd", "RegWr"};
    for (size_t i = 0; i < m_test_jobs.size(); i++) {
      std::vector<std::string> f;
      std::stringstream ss(m_test_jobs[i]);
      for (std::string item; std::getline(ss, item, ':');) {
        f.push_back(item);
      }
      if (f.size() != 4) {
        throw std::runtime_error("PimSequencer: test job '" + m_test_jobs[i] + "' is not <clk>:<class>:<addr>:<ncols>");
      }
      TestJob t;
      t.at = std::stoll(f[0]);
      int cls = -1;
      for (int c = 0; c < static_cast<int>(PimClass::COUNT); c++) {
        if (f[1] == names[c]) {
          cls = c;
        }
      }
      if (cls < 0) {
        throw std::runtime_error("PimSequencer: test job class '" + f[1] + "' is unknown");
      }
      t.job.cls = static_cast<PimClass>(cls);
      std::stringstream as(f[2]);
      for (std::string a; std::getline(as, a, '/');) {
        t.job.addr.push_back(std::stoi(a));
      }
      t.job.addr[0] = m_ctrl->m_channel_id;
      t.job.ncols = std::stoi(f[3]);
      t.job.id = i;
      m_tests.push_back(t);
    }
    s_test_last.assign(m_tests.size(), 0);
    s_test_done.assign(m_tests.size(), 0);
    for (size_t i = 0; i < m_tests.size(); i++) {
      m_stats.add(fmt::format("pim_test_last_cmd_{}", i), s_test_last[i]);
      m_stats.add(fmt::format("pim_test_done_{}", i), s_test_done[i]);
    }
  }

  void run_test_inputs(Clk_t clk) {
    for (size_t i = 0; i < m_tests.size(); i++) {
      auto& t = m_tests[i];
      if (t.submitted || clk < t.at) {
        continue;
      }
      t.submitted = true;
      t.job.accept_clk = clk;
      submit(
          t.job, [this](uint64_t id) { s_test_last[id] = m_ctrl->m_clk; },
          [this](uint64_t id, Clk_t done) { s_test_done[id] = done; });
    }
    if (m_test_write_drain.size() == 2) {
      if (clk == m_test_write_drain[0]) {
        m_ctrl->request_write_drain(true);
        note_write_drain(true);
      } else if (clk == m_test_write_drain[1]) {
        m_ctrl->request_write_drain(false);
        note_write_drain(false);
      }
    }
  }
};

}  // namespace Ramulator
