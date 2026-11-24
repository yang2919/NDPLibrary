// Exercise the public L2 entry points against a separate FP16 libpim worker.
#include "ndp/l1/gem5_pim/backend.hpp"
#include "ndp/l3/weight.hpp"
#include "test_support.hpp"

int main(int argc, char **argv) {
  using namespace ndp;
  CHECK(argc == 2);
  auto backend = make_gem5_pim_backend(argv[1]);
  CHECK(backend);
  auto created = Session::create(std::move(backend.value()));
  CHECK(created);
  auto session = std::move(created.value());
  auto w = reference::upload(session, {2, 3}, {1, 2, 3, -1, 0, 2});
  auto x = reference::upload(session, {3}, {1, 1, 1});
  OpOptions prepared{Engine::PIM, LayoutPolicy::RequirePrepared};
  CHECK(!session.matvec(w, x, prepared));
  CHECK(!prepare_weight(session, w, {{Engine::PIM}, 12}));
  auto weight = prepare_weight(session, w, {{Engine::PIM}, 1 << 20});
  CHECK(weight);
  auto back = session.to_host(weight.value().tensor(), LayoutKind::PimBanked);
  CHECK(back && back.value().bytes == session.to_host(w).value().bytes);
  auto y = session.matvec(weight.value().tensor(), x, prepared);
  CHECK(y);
  auto projection = reference::upload(session, {1, 2}, {1, -1});
  auto chained = session.matvec(projection, y.value(), {Engine::PIM});
  CHECK(chained);
  reference::close(reference::download(session, chained.value()), {5}, 0, 0);
  auto next = reference::upload(session, {1, 3}, {2, 0, -1});
  auto z = session.linear(next, weight.value().tensor(), prepared);
  CHECK(z);
  reference::close(reference::download(session, z.value()), {-1, -4}, 0, 0);
  CHECK(!session.matvec(w, x, {Engine::PNM}));
  CHECK(!session.add(x, x, {Engine::PIM}));
  CHECK(session.close());
}
