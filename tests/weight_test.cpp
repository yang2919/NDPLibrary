// Verify L3 preparation policy, budget enforcement, reuse, and failure isolation through L2.
#include "ndp/l3/weight.hpp"
#include "test_support.hpp"

using namespace ndp;

struct Probe {
  int packs = 0;
  bool fail = false;
};

class PackingBackend : public reference::ForwardBackend {
public:
  explicit PackingBackend(std::shared_ptr<Probe> probe) : probe_(std::move(probe)) {}

  Result<void> transform(AllocationHandle a, const LayoutDesc &ad, AllocationHandle b,
                         const LayoutDesc &bd) override {
    ++probe_->packs;
    if (probe_->fail) {
      return Status{ErrorCode::BackendFailure, "Injected packing failure"};
    }
    return inner->transform(a, ad, b, bd);
  }

private:
  std::shared_ptr<Probe> probe_;
};

void preparation() {
  auto probe = std::make_shared<Probe>();
  auto made = Session::create(std::make_unique<PackingBackend>(probe));
  CHECK(made);
  auto s = std::move(made.value());
  auto w = reference::upload(s, {2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK(w.wait());
  const auto baseline = s.memory_stats().live_bytes;
  auto short_budget = prepare_weight(s, w, {{Engine::PNM, Engine::PIM}, 43});
  CHECK(!short_budget && short_budget.error().code == ErrorCode::OutOfMemory);
  CHECK(s.memory_stats().live_bytes == baseline && probe->packs == 0);
  CHECK(!prepare_weight(s, w, {{}, 44}));
  CHECK(!prepare_weight(s, w, {{static_cast<Engine>(99)}, 44}));
  auto vector = reference::upload(s, {3}, {1, 2, 3});
  CHECK(!prepare_weight(s, vector));
  auto pnm = prepare_weight(s, w, {{Engine::PNM}, 12});
  CHECK(pnm && pnm.value().resident_bytes() == 12 && probe->packs == 0);
  auto both = prepare_weight(s, w, {{Engine::PNM, Engine::PIM, Engine::PIM}, 44});
  CHECK(both && both.value().resident_bytes() == 44 && probe->packs == 1);
  CHECK(w.representations().size() == 1);
  auto reused = prepare_weight(s, both.value().tensor(), {{Engine::PIM}, 44});
  CHECK(reused && probe->packs == 1);
  CHECK(!prepare_weight(s, both.value().tensor(), {{Engine::PNM}, 12}));
  auto x = reference::upload(s, {1, 3}, {1, 2, 3});
  for (auto engine : {Engine::PNM, Engine::PIM}) {
    auto y = s.linear(x, both.value().tensor(), {engine, LayoutPolicy::RequirePrepared});
    CHECK(y);
    reference::close(reference::download(s, y.value()), {14, 32}, 0, 0);
  }
  auto other = Session::create(make_functional_backend(4096));
  CHECK(other);
  CHECK(!prepare_weight(other.value(), w));
  CHECK(s.close());
  CHECK(!prepare_weight(s, w));
}

void failure_isolation() {
  auto probe = std::make_shared<Probe>();
  probe->fail = true;
  auto made = Session::create(std::make_unique<PackingBackend>(probe));
  CHECK(made);
  auto s = std::move(made.value());
  auto w = reference::upload(s, {2, 3}, {1, 2, 3, 4, 5, 6});
  auto failed = prepare_weight(s, w);
  CHECK(!failed);
  CHECK(w.representations().size() == 1);
  CHECK(s.memory_stats().live_bytes == 12);
  reference::close(reference::download(s, w), {1, 2, 3, 4, 5, 6}, 0, 0);
  probe->fail = false;
  auto retry = prepare_weight(s, w);
  CHECK(retry);
  CHECK(retry.value().resident_bytes() == 44);
  CHECK(!s.close());
}

int main() {
  try {
    preparation();
    failure_isolation();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "L3 weight contracts passed\n";
}
