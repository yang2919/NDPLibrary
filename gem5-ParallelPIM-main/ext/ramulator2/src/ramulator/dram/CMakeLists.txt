add_library(ramulator-dram OBJECT)

target_sources(
  ramulator-dram PRIVATE
  node.h  node.cpp
  device.h  device.cpp
  dram_spec.h  dram_spec.cpp
  func_types.h

  commands/ACT.h  commands/ACT1.h  commands/ACT2.h
  commands/PREpb.h  commands/PREab.h  commands/PREsb.h
  commands/RD.h   commands/WR.h
  commands/RDA.h  commands/WRA.h
  commands/RD_S.h   commands/WR_S.h
  commands/RDA_S.h  commands/WRA_S.h
  commands/RD_L.h   commands/WR_L.h
  commands/RDA_L.h  commands/WRA_L.h
  commands/REFab.h  commands/REFpb.h  commands/REFsb.h
  commands/RFMab.h  commands/RFMpb.h  commands/RFMsb.h
  commands/RCKSTRT.h  commands/RCKSTOP.h
  commands/CAS.h  commands/CAS_RD.h  commands/CAS_WR.h
  commands/VRR.h
  commands/ACTab.h  commands/ACTab1.h  commands/ACTab2.h
  commands/ACTsb1.h  commands/ACTsb2.h
  commands/PIM_SB_RD.h  commands/PIM_SB_RD2.h  commands/PIM_SB_WR.h
  commands/PIM_SB_RDA.h  commands/PIM_SB_RD2A.h  commands/PIM_SB_WRA.h
  commands/PIM_AB_EW.h  commands/PIM_AB_MAC.h
  commands/PIM_AB_BKREG.h  commands/PIM_AB_REGBK.h
  commands/PIM_GB_RD.h  commands/PIM_GB_WR.h
  commands/PIM_REG_RD.h  commands/PIM_REG_WR.h
  commands/populate.h

  impl/DDR3.cpp
  impl/DDR4.cpp
  impl/DDR4_VRR.cpp
  impl/DDR5.cpp
  impl/DDR5_VRR.cpp
  impl/HBM1.cpp
  impl/HBM2.cpp
  impl/HBM3.cpp
  impl/HBM4.cpp
  impl/LPDDR5.cpp
  impl/LPDDR6.cpp
  impl/GDDR6.cpp
  impl/GDDR7.cpp
  impl/DDR5_PIM.cpp
  impl/LPDDR6_PIM.cpp
  impl/GDDR7_PIM.cpp
  impl/HBM4_PIM.cpp
)

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-dram
)
