#ifndef RAMULATOR_PIM_PIM_ARB_H
#define RAMULATOR_PIM_PIM_ARB_H

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

// gem5's processing-in-memory (PIM) arbitration policies, kept as pure code.
// The sequencer fills the view each cycle
// and asks whether a PIM command may go before host work.

namespace Ramulator {

enum class PimSide { Host, Pim };

struct PimArbView {
  bool pim_pending = false;  // a job is queued in the unit (PIM work)
  bool pim_ready = false;    // head job past Hazard (submitted), next command legal
  bool pim_started = false;  // head job has issued a command
  bool pim_row_hit = false;  // rows of its next command open (non-bank: true)
  bool host_queued = false;  // host reads or writes waiting (and the wrapper's backlog)
  bool host_ready = false;   // the controller would pick a host request now
  bool host_row_hit = false;
  bool host_older = false;  // that request arrived before the PIM job
  PimSide last_side = PimSide::Host;
  bool last_was_pre = false;  // the last command was a PRE (or closed a row)
};

// When the sequencer stops new host work in a job's unit.
enum class PimGate {
  WhileStarted,   // from the job's first to its last command (FR-FCFS)
  WhileSelected,  // while the policy selects the PIM side (AsyncDIMM)
  Never,          // Chopim
};

class PimArbPolicy {
 public:
  virtual ~PimArbPolicy() = default;
  virtual PimSide select(const PimArbView& v) const = 0;
  virtual PimGate gate() const = 0;
};

// First-ready, first-come, first-served: a started job is never interrupted;
// otherwise the ready side wins, then the row hit, then the older request.
class FrfcfsPolicy final : public PimArbPolicy {
 public:
  PimSide select(const PimArbView& v) const override {
    if (v.pim_started) {
      return PimSide::Pim;
    }
    if (!v.pim_ready) {
      return PimSide::Host;
    }
    if (!v.host_ready) {
      return PimSide::Pim;
    }
    if (v.pim_row_hit != v.host_row_hit) {
      return v.pim_row_hit ? PimSide::Pim : PimSide::Host;
    }
    return v.host_older ? PimSide::Host : PimSide::Pim;
  }
  PimGate gate() const override {
    return PimGate::WhileStarted;
  }
};

// Chopim: host first. A PIM command only when no host request is queued,
// before and during a job; no gate. Held
// packets do not count, so this cannot deadlock.
class ChopimPolicy final : public PimArbPolicy {
 public:
  PimSide select(const PimArbView& v) const override {
    return v.host_queued ? PimSide::Host : PimSide::Pim;
  }
  PimGate gate() const override {
    return PimGate::Never;
  }
};

// AsyncDIMM: stay on the last side while it has work;
// switch after a PRE of the current side if the other side has work, or
// when the current side has none. The first side is the host; on the PIM
// side the unit is gated.
class AsyncDimmPolicy final : public PimArbPolicy {
 public:
  PimSide select(const PimArbView& v) const override {
    bool pim_side = v.last_side == PimSide::Pim;
    bool current = pim_side ? v.pim_pending : v.host_queued;
    bool other = pim_side ? v.host_queued : v.pim_pending;
    if (!current || (v.last_was_pre && other)) {
      pim_side = !pim_side;
    }
    return pim_side ? PimSide::Pim : PimSide::Host;
  }
  PimGate gate() const override {
    return PimGate::WhileSelected;
  }
};

// COSM is deferred until it is defined: choosing it stops the run.
class CosmPolicy final : public PimArbPolicy {
 public:
  CosmPolicy() {
    throw std::runtime_error("arbitration policy cosm is pending its definition");
  }
  PimSide select(const PimArbView& v) const override {
    return PimSide::Host;
  }
  PimGate gate() const override {
    return PimGate::Never;
  }
};

inline std::unique_ptr<PimArbPolicy> make_pim_arb_policy(const std::string& name) {
  if (name == "frfcfs") {
    return std::make_unique<FrfcfsPolicy>();
  }
  if (name == "chopim") {
    return std::make_unique<ChopimPolicy>();
  }
  if (name == "asyncdimm") {
    return std::make_unique<AsyncDimmPolicy>();
  }
  if (name == "cosm") {
    return std::make_unique<CosmPolicy>();
  }
  throw std::runtime_error("PimSequencer: unknown arb_policy '" + name +
                           "' (expected frfcfs, chopim, asyncdimm or cosm)");
}

}  // namespace Ramulator

#endif  // RAMULATOR_PIM_PIM_ARB_H
