#include "ramulator/dram/device.h"

#include <stdexcept>

namespace Ramulator {

void DRAMDevice::init(std::unique_ptr<DRAMSpec> spec) {
  m_spec_owner = std::move(spec);
  m_spec = m_spec_owner.get();
  m_bank_level = m_spec->get_level_id("Bank");
  m_root = std::make_unique<DRAMNode>(m_spec, nullptr, 0, 0);
  m_root->for_each_at_level(m_bank_level, [&](DRAMNode* bank) { m_bank_nodes.push_back(bank); });
}

void DRAMDevice::set_channel_id(int channel_id) {
  m_root->m_node_id = channel_id;
}

void DRAMDevice::issue_command(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  if (m_spec->bank_targets[command] == BankTarget::Pair) {
    // gem5 (README.gem5 change 7): timing is hierarchical by address, so a pair command
    // updates the path of both banks.
    m_root->update_timing(command, pair_addr_vec(addr_vec, 0), clk);
    m_root->update_timing(command, pair_addr_vec(addr_vec, 1), clk);
  } else {
    m_root->update_timing(command, addr_vec, clk);
  }
  apply_action(command, addr_vec, clk);
}

bool DRAMDevice::check_timing(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  if (m_spec->bank_targets[command] == BankTarget::Pair) {
    // gem5 (README.gem5 change 7): both banks of the pair must allow the command.
    return m_root->check_timing(command, pair_addr_vec(addr_vec, 0), clk) &&
           m_root->check_timing(command, pair_addr_vec(addr_vec, 1), clk);
  }
  return m_root->check_timing(command, addr_vec, clk);
}

AddrVec_t DRAMDevice::pair_addr_vec(const AddrVec_t& addr_vec, int half) const {
  AddrVec_t a = addr_vec;
  if (a[m_bank_level] < 0 || m_spec->organization.level_sizes[m_bank_level] < 2) {
    throw std::runtime_error("DRAMDevice: a Pair command needs a concrete bank and at least two banks per group");
  }
  a[m_bank_level] = (a[m_bank_level] & ~1) | half;
  return a;
}

int DRAMDevice::get_preq_command(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto preq_fn = m_spec->funcs.preqs[command];
  if (!preq_fn) return command;

  int resolved = command;
  for_each_target_bank_while(command, addr_vec, [&](int flat_bank_id) {
    int preq = preq_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
    if (preq != command) { resolved = preq; return false; }
    return true;
  });
  return resolved;
}

bool DRAMDevice::check_rowbuffer_hit(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto rowhit_fn = m_spec->funcs.rowhits[command];
  if (!rowhit_fn) {
    return false;
  }
  int flat_bank_id = get_flat_bank_id(addr_vec);
  return rowhit_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
}

bool DRAMDevice::check_node_open(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto rowopen_fn = m_spec->funcs.rowopens[command];
  if (!rowopen_fn) {
    return false;
  }
  int flat_bank_id = get_flat_bank_id(addr_vec);
  return rowopen_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
}

int DRAMDevice::get_flat_bank_id(const AddrVec_t& addr_vec) const {
  int id = 0;
  for (int lvl = 1; lvl <= m_bank_level; lvl++) {
    id = id * m_spec->organization.level_sizes[lvl] + addr_vec[lvl];
  }
  return id;
}

bool DRAMDevice::bank_matches(DRAMNode* bank, const AddrVec_t& addr_vec) {
  for (auto* n = bank; n != nullptr; n = n->m_parent_node) {
    if (addr_vec[n->m_level] != -1 && addr_vec[n->m_level] != n->m_node_id) {
      return false;
    }
  }
  return true;
}

std::vector<int> DRAMDevice::get_target_banks(int command, const AddrVec_t& addr_vec) const {
  std::vector<int> ids;
  for_each_target_bank(command, addr_vec, [&](int id) { ids.push_back(id); });
  return ids;
}

void DRAMDevice::apply_action(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto action_fn = m_spec->funcs.actions[command];
  if (!action_fn) return;
  for_each_target_bank(command, addr_vec, [&](int flat_bank_id) {
    action_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
  });
}

}  // namespace Ramulator
