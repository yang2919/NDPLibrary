add_library(ramulator-pim OBJECT)

target_sources(
  ramulator-pim PRIVATE
  pim_arb.h
  pim_unit.h
  pim_sequencer.cpp
)

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-pim
)
