#include <fmt/format.h>
#include <stdexcept>

#include "ramulator/base/param.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/memory_system/channel_mapper/i_channel_mapper.h"
#include "ramulator/memory_system/i_memory_system.h"
#include "ramulator/memory_system/i_pim_memory_system.h"
#include "ramulator/pim/pim_unit.h"
#include "ramulator/translation/i_translation.h"

namespace Ramulator {

// gem5's processing-in-memory (PIM) memory system:
// GenericDRAMSystem (generic_dram_system.cpp, copied because it is final)
// plus the entry points gem5's wrapper uses to reach each channel's PIM
// sequencer, the decode and the write drain. Without PIM requests it
// behaves as GenericDRAM.
class PimDRAMSystem final : public IMemorySystem, public IPimMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, PimDRAMSystem, "PimDRAM");

 protected:
  IChannelMapper* m_channel_mapper;
  std::vector<IController*> m_controllers;
  unsigned int m_clock_ratio = 1;
  int m_tx_bytes = 0;

  // gem5 PIM
  std::vector<ControllerBase*> m_bases;
  std::vector<IPimSequencer*> m_sequencers;
  std::function<void(int, Addr_t, bool)> m_host_column_listener;
  bool m_checked = false;

 public:
  int s_num_read_requests = 0;
  int s_num_write_requests = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_CREATE_CHILD(m_channel_mapper, IChannelMapper);

    // Each controller = one channel. DRAM config lives inside each controller.
    RAMULATOR_CREATE_CHILD_LIST(m_controllers, IController);
    if (m_controllers.empty()) {
      throw std::runtime_error("PimDRAM requires at least one controller");
    }
    for (size_t i = 0; i < m_controllers.size(); i++) {
      dynamic_cast<Implementation*>(m_controllers[i])->set_id(fmt::format("Channel {}", i));
      m_controllers[i]->set_channel_id(static_cast<int>(i));
      m_controllers[i]->m_clock_ratio = m_clock_ratio;
      auto* base = dynamic_cast<ControllerBase*>(m_controllers[i]);
      if (!base) {
        throw std::runtime_error("PimDRAM requires ControllerBase-derived controllers");
      }
      m_bases.push_back(base);
    }
    m_sequencers.assign(m_controllers.size(), nullptr);

    // Setup channel mapper with controller info
    m_tx_bytes = m_controllers[0]->get_tx_bytes();
    m_channel_mapper->setup(static_cast<int>(m_controllers.size()), calc_log2(m_tx_bytes));

    m_stats.add("total_num_read_requests", s_num_read_requests);
    m_stats.add("total_num_write_requests", s_num_write_requests);
  };

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
  }

  bool send(Request& req) override {
    // Validate request size: must be set and fit within one transaction.
    if (req.size_bytes <= 0 || req.size_bytes > m_tx_bytes) {
      throw std::runtime_error(fmt::format("Request size_bytes must be set by the frontend (got {}, tx_bytes = {}).",
                                           req.size_bytes, m_tx_bytes));
    }

    // Channel mapper sets req.addr_vec[0] and req.intra_channel_addr.
    // Controller::send() handles address mapping internally.
    m_channel_mapper->apply(req);
    int channel_id = req.addr_vec[0];
    bool is_success = m_controllers[channel_id]->send(req);

    if (is_success) {
      switch (req.type_id) {
        case Request::Type::Read: {
          s_num_read_requests++;
          break;
        }
        case Request::Type::Write: {
          s_num_write_requests++;
          break;
        }
      }
    }
    return is_success;
  };

  void tick() override {
    if (!m_checked) {
      // Every sequencer registered itself in setup() (connect_frontend).
      for (size_t i = 0; i < m_sequencers.size(); i++) {
        if (!m_sequencers[i]) {
          throw std::runtime_error(fmt::format("PimDRAM: channel {} has no PimSequencer controller plugin", i));
        }
      }
      m_checked = true;
    }
    for (auto controller : m_controllers) {
      controller->tick();
    }
  };

  void reset_stats() override {
    s_num_read_requests = 0;
    s_num_write_requests = 0;
  }

  int get_clock_ratio() override {
    return m_clock_ratio;
  }

  float get_tCK() override {
    if (!m_controllers.empty()) {
      return m_controllers[0]->get_tCK();
    }
    return -1.0f;
  }

  int get_tx_bytes() override {
    return m_tx_bytes;
  }

  // ── IPimMemorySystem ───────────────────────────────────────────────────

  AddrVec_t decode(int port, Addr_t offset) override {
    Request req(offset, Request::Type::Read);
    req.ingress_id = port;
    m_channel_mapper->apply(req);
    int channel = req.addr_vec[0];
    m_bases.at(channel)->m_addr_mapper->apply(req);
    req.addr_vec[0] = channel;
    return req.addr_vec;
  }

  PimGeometry geometry() const override {
    const auto& spec = *m_bases[0]->m_device.m_spec;
    PimGeometry g;
    g.level_names = spec.level_names;
    for (int l = 0; l < spec.level_count; l++) {
      g.level_counts.push_back(spec.organization.level_sizes[l]);
    }
    g.level_counts.back() /= spec.internal_prefetch_size;  // the column counts transactions (F16)
    g.channels = static_cast<int>(m_controllers.size());
    g.tx_bytes = m_tx_bytes;
    g.unit_level = pim_unit_level(spec);
    return g;
  }

  IPimSequencer* sequencer(int channel) override {
    return m_sequencers.at(channel);
  }

  void register_sequencer(int channel, IPimSequencer* s) override {
    if (m_sequencers.at(channel) && m_sequencers[channel] != s) {
      throw std::runtime_error(fmt::format("PimDRAM: channel {} has two PimSequencer plugins", channel));
    }
    m_sequencers[channel] = s;
  }

  Clk_t clk(int channel) const override {
    return m_controllers.at(channel)->m_clk;
  }

  void request_write_drain(int channel, bool on) override {
    m_bases.at(channel)->request_write_drain(on);
    if (m_sequencers[channel]) {
      m_sequencers[channel]->note_write_drain(on);
    }
  }

  size_t read_buffer_size(int channel) const override {
    return m_bases.at(channel)->read_buffer_size();
  }

  size_t read_buffer_capacity(int channel) const override {
    return m_bases.at(channel)->read_buffer_capacity();
  }

  size_t write_buffer_size(int channel) const override {
    return m_bases.at(channel)->write_buffer_size();
  }

  size_t write_buffer_capacity(int channel) const override {
    return m_bases.at(channel)->write_buffer_capacity();
  }

  PimTimingInfo timing_info(int channel) const override {
    const auto& spec = *m_bases.at(channel)->m_device.m_spec;
    PimTimingInfo t;
    t.standard = spec.standard_name;
    t.level_names = spec.level_names;
    t.level_sizes = spec.organization.level_sizes;
    t.command_names = spec.command_names;
    t.command_cycles = spec.command_cycles;
    for (const auto& m : spec.command_meta) {
      t.is_closing.push_back(m.is_closing);
    }
    for (BankTarget b : spec.bank_targets) {
      t.is_pair.push_back(b == BankTarget::Pair);
    }
    for (const auto& lv : spec.has_sibling_cons) {
      t.has_sibling_cons.emplace_back(lv.begin(), lv.end());
    }
    for (int l = 0; l < static_cast<int>(spec.timing_cons.size()); l++) {
      for (int c = 0; c < static_cast<int>(spec.timing_cons[l].size()); c++) {
        for (const auto& e : spec.timing_cons[l][c]) {
          t.entries.push_back({l, c, e.cmd, e.val, e.window, e.sibling, e.history_group});
        }
      }
    }
    t.timing_names = spec.timing_names;
    t.timing_vals = spec.timing_vals;
    if (m_sequencers.at(channel)) {
      m_sequencers[channel]->describe(t);
    }
    return t;
  }

  void set_host_column_listener(std::function<void(int, Addr_t, bool)> f) override {
    m_host_column_listener = std::move(f);
  }

  void host_column_issued(int channel, Addr_t addr, bool is_read) override {
    if (m_host_column_listener) {
      m_host_column_listener(channel, addr, is_read);
    }
  }
};

}  // namespace Ramulator
