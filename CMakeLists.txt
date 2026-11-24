# Build the NDP library, runnable examples, and contract test suites.
cmake_minimum_required(VERSION 3.20)
project(ndp_library LANGUAGES CXX)

find_package(Threads REQUIRED)

add_library(ndp
  src/common/compute.cpp
  src/common/fp16.cpp
  src/common/tensor_metadata.cpp
  src/l1/functional/backend.cpp
  src/l1/ndp/transport.cpp
  src/l1/ndp/codec.cpp
  src/l1/ndp/emulator.cpp
  src/l1/ndp/command_builder.cpp
  src/l1/ndp/backend.cpp
  src/l1/ndp/program.cpp
  src/l1/ndp/launch.cpp
  src/l1/ndp/pnm.cpp
  src/l1/ndp/config.cpp
  src/l1/common/functional_ops.cpp
  src/l2/device_group.cpp
  src/l2/session.cpp
)
target_include_directories(ndp PUBLIC include)
target_compile_features(ndp PUBLIC cxx_std_20)
target_link_libraries(ndp PRIVATE Threads::Threads)
target_compile_options(ndp PRIVATE -Wall -Wextra -Wpedantic)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_source_files_properties(src/l1/ndp/emulator.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()

# L3 depends on L2; the core library has no cache/model dependency.
add_library(ndp_llm src/l3/llm.cpp src/l3/decoder.cpp src/l3/weight.cpp)
target_link_libraries(ndp_llm PUBLIC ndp)
target_compile_options(ndp_llm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(single_device_linear examples/single_device_linear.cpp)
target_link_libraries(single_device_linear PRIVATE ndp)

add_executable(decoder_block examples/decoder_block.cpp)
target_link_libraries(decoder_block PRIVATE ndp_llm)

# The metadata-only probe uses native Unix processes and never executes commands.
if(UNIX)
  add_library(ndp_gem5_pim STATIC src/l1/gem5_pim/backend.cpp)
  target_link_libraries(ndp_gem5_pim PUBLIC ndp)
  target_compile_options(ndp_gem5_pim PRIVATE -Wall -Wextra -Wpedantic)
  add_library(ndp_metadata STATIC src/l1/gem5/metadata.cpp)
  target_include_directories(ndp_metadata PUBLIC src)
  target_link_libraries(ndp_metadata PUBLIC ndp)
  target_compile_options(ndp_metadata PRIVATE -Wall -Wextra -Wpedantic)
  add_library(ndp_execution STATIC src/l1/gem5/execution.cpp)
  target_link_libraries(ndp_execution PUBLIC ndp_metadata)
  target_compile_options(ndp_execution PRIVATE -Wall -Wextra -Wpedantic)
  add_library(ndp_gem5 STATIC src/l1/gem5/backend.cpp)
  target_link_libraries(ndp_gem5 PUBLIC ndp_execution Threads::Threads)
  target_compile_options(ndp_gem5 PRIVATE -Wall -Wextra -Wpedantic)
  add_executable(ndp_execution_receiver tools/gem5/guest/execution_receiver.cpp)
  target_link_libraries(ndp_execution_receiver PRIVATE ndp_execution)
  add_executable(ndp_decoder_probe tools/gem5/guest/decoder_probe.cpp)
  target_link_libraries(ndp_decoder_probe PRIVATE ndp_gem5 ndp_llm)
  add_executable(ndp_session_probe tools/gem5/guest/session_probe.cpp)
  target_link_libraries(ndp_session_probe PRIVATE ndp_gem5)
  add_executable(ndp_execution_probe tools/gem5/guest/execution_probe.cpp)
  target_link_libraries(ndp_execution_probe PRIVATE ndp_execution)
  add_executable(ndp_metadata_probe tools/gem5/guest/metadata_probe.cpp)
  target_link_libraries(ndp_metadata_probe PRIVATE ndp_metadata)
endif()

# Optional native conformance exporter; CENT/Python remain external dependencies.
add_executable(ndp_cent_export tools/cent/export_case.cpp)
target_include_directories(ndp_cent_export PRIVATE src)
target_link_libraries(ndp_cent_export PRIVATE ndp)

# Examples remain available when contract tests are disabled.
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(UNIX AND Python3_Interpreter_FOUND)
    add_executable(ndp_pnm_test_worker tools/gem5/pnm/service.cpp)
    target_include_directories(ndp_pnm_test_worker PRIVATE src)
    target_link_libraries(ndp_pnm_test_worker PRIVATE ndp)
    add_test(NAME pnm_worker_protocol COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/gem5/pnm/test_service.py $<TARGET_FILE:ndp_pnm_test_worker>)
    set_tests_properties(pnm_worker_protocol PROPERTIES TIMEOUT 20)
  endif()
  # CENT is opt-in; default builds never download it or require PyTorch.
  find_package(Python3 QUIET COMPONENTS Interpreter)
  set(NDP_CENT_SOURCE "" CACHE FILEPATH "Pinned CENT cent_simulation/aim_sim.py for replay tests")
  if(UNIX AND Python3_Interpreter_FOUND)
    add_test(NAME cent_codec_contracts COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/cent/test_codec.py)
    if(NDP_CENT_SOURCE)
      if(NOT EXISTS "${NDP_CENT_SOURCE}")
        message(FATAL_ERROR "NDP_CENT_SOURCE does not exist")
      endif()
      add_test(NAME cent_replay_contracts COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cent/test_replay.py
        $<TARGET_FILE:ndp_cent_export> ${NDP_CENT_SOURCE})
      set_tests_properties(cent_replay_contracts PROPERTIES TIMEOUT 60)
    endif()
  endif()
  if(UNIX)
    set(NDP_GEM5_PIM_WORKER "" CACHE FILEPATH "Optional native FP16 libpim worker for Session tests")
    if(NDP_GEM5_PIM_WORKER)
      add_executable(ndp_gem5_pim_tests tests/gem5_pim_test.cpp)
      target_link_libraries(ndp_gem5_pim_tests PRIVATE ndp_gem5_pim ndp_llm)
      add_test(NAME gem5_pim_session COMMAND ndp_gem5_pim_tests "${NDP_GEM5_PIM_WORKER}")
      set_tests_properties(gem5_pim_session PROPERTIES TIMEOUT 30)
    endif()
    add_executable(ndp_metadata_tests tests/metadata_test.cpp)
    target_link_libraries(ndp_metadata_tests PRIVATE ndp_metadata)
    add_executable(ndp_execution_tests tests/execution_test.cpp)
    target_link_libraries(ndp_execution_tests PRIVATE ndp_execution Threads::Threads)
    add_test(NAME gem5_decoder COMMAND ndp_decoder_probe $<TARGET_FILE:ndp_execution_receiver>)
    set_tests_properties(gem5_decoder PROPERTIES TIMEOUT 60)
    add_test(NAME gem5_session COMMAND ndp_session_probe $<TARGET_FILE:ndp_execution_receiver>)
    set_tests_properties(gem5_session PROPERTIES TIMEOUT 30)
    add_test(NAME execution_protocol COMMAND ndp_execution_tests)
    set_tests_properties(execution_protocol PROPERTIES TIMEOUT 20)
    add_test(NAME execution_processes COMMAND ndp_execution_probe)
    set_tests_properties(execution_processes PROPERTIES TIMEOUT 40)
    add_test(NAME metadata_framing COMMAND ndp_metadata_tests)
    add_test(NAME metadata_processes COMMAND ndp_metadata_probe)
    set_tests_properties(metadata_framing metadata_processes PROPERTIES TIMEOUT 40)
  endif()
  add_executable(ndp_transport_tests tests/ndp_transport_test.cpp)
  target_include_directories(ndp_transport_tests PRIVATE src)
  target_link_libraries(ndp_transport_tests PRIVATE ndp)
  add_test(NAME ndp_transport_contracts COMMAND ndp_transport_tests)
  set_tests_properties(ndp_transport_contracts PROPERTIES TIMEOUT 20)
  add_executable(ndp_command_tests tests/ndp_command_test.cpp)
  target_include_directories(ndp_command_tests PRIVATE src)
  target_link_libraries(ndp_command_tests PRIVATE ndp)
  add_test(NAME ndp_command_contracts COMMAND ndp_command_tests)
  set_tests_properties(ndp_command_contracts PROPERTIES TIMEOUT 20)
  add_executable(ndp_backend_tests tests/ndp_backend_test.cpp)
  target_include_directories(ndp_backend_tests PRIVATE src)
  target_link_libraries(ndp_backend_tests PRIVATE ndp)
  add_test(NAME ndp_backend_contracts COMMAND ndp_backend_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/ndp-small.ini)
  set_tests_properties(ndp_backend_contracts PROPERTIES TIMEOUT 20)
  add_executable(ndp_program_tests tests/ndp_program_test.cpp)
  target_include_directories(ndp_program_tests PRIVATE src)
  target_link_libraries(ndp_program_tests PRIVATE ndp)
  add_test(NAME ndp_program_contracts COMMAND ndp_program_tests)
  set_tests_properties(ndp_program_contracts PROPERTIES TIMEOUT 20)
  add_executable(ndp_pnm_tests tests/ndp_pnm_test.cpp)
  target_include_directories(ndp_pnm_tests PRIVATE src)
  target_link_libraries(ndp_pnm_tests PRIVATE ndp)
  add_test(NAME ndp_pnm_contracts COMMAND ndp_pnm_tests)
  set_tests_properties(ndp_pnm_contracts PROPERTIES TIMEOUT 20)
  add_test(NAME ndp_decoder_example COMMAND decoder_block --ndp-config
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/ndp-small.ini)
  add_test(NAME ndp_decoder_default COMMAND decoder_block --ndp)
  set_tests_properties(ndp_decoder_example ndp_decoder_default PROPERTIES TIMEOUT 20)
  foreach(name session async layout compute buffer)
    add_executable(ndp_${name}_tests tests/${name}_test.cpp)
    target_link_libraries(ndp_${name}_tests PRIVATE ndp)
    add_test(NAME ${name}_contracts COMMAND ndp_${name}_tests)
    set_tests_properties(${name}_contracts PROPERTIES TIMEOUT 20)
  endforeach()

  foreach(name llm cache decoder weight)
    add_executable(ndp_${name}_tests tests/${name}_test.cpp)
    target_link_libraries(ndp_${name}_tests PRIVATE ndp_llm)
    add_test(NAME ${name}_contracts COMMAND ndp_${name}_tests)
    set_tests_properties(${name}_contracts PROPERTIES TIMEOUT 20)
  endforeach()
endif()

# Python/Torch dependencies are confined to an explicitly enabled adapter target.
option(NDP_BUILD_PYTHON "Build the PyTorch Python adapter" OFF)
option(NDP_PYTHON_TESTING "Include Python adapter fault-injection tests" OFF)
if(NDP_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE NDP_TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  list(APPEND CMAKE_PREFIX_PATH "${NDP_TORCH_PREFIX}")
  find_package(Torch REQUIRED)
  find_library(NDP_TORCH_PYTHON torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
  set_target_properties(ndp ndp_llm PROPERTIES POSITION_INDEPENDENT_CODE ON)
  separate_arguments(NDP_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  # Match libtorch's C++ ABI for all targets crossing the adapter boundary.
  target_compile_options(ndp PUBLIC ${NDP_TORCH_FLAGS})
  Python3_add_library(ndp_python MODULE WITH_SOABI
    bindings/pytorch/module.cpp bindings/pytorch/session.cpp bindings/pytorch/tensor.cpp
    bindings/pytorch/operations.cpp bindings/pytorch/llm.cpp bindings/pytorch/dispatch.cpp)
  if(NDP_PYTHON_TESTING)
    target_sources(ndp_python PRIVATE tests/python/failing_backend.cpp)
    target_compile_definitions(ndp_python PRIVATE NDP_PYTHON_TESTING)
  endif()
  target_link_libraries(ndp_python PRIVATE ndp_llm ${TORCH_LIBRARIES} ${NDP_TORCH_PYTHON})
  if(TARGET ndp_gem5_pim)
    set_target_properties(ndp_gem5_pim PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ndp_python PRIVATE ndp_gem5_pim)
    target_compile_definitions(ndp_python PRIVATE NDP_PYTHON_GEM5_PIM)
  endif()
  target_include_directories(ndp_python PRIVATE ${TORCH_INCLUDE_DIRS})
  set_target_properties(ndp_python PROPERTIES OUTPUT_NAME _C)
  if(NOT DEFINED NDP_PYTHON_OUTPUT_DIR)
    set(NDP_PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/ndp")
  endif()
  set_target_properties(ndp_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${NDP_PYTHON_OUTPUT_DIR}")
  file(MAKE_DIRECTORY "${NDP_PYTHON_OUTPUT_DIR}")
  file(GLOB NDP_PYTHON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/ndp/*.py")
  foreach(source IN LISTS NDP_PYTHON_SOURCES)
    get_filename_component(name "${source}" NAME)
    if(NOT "${source}" STREQUAL "${NDP_PYTHON_OUTPUT_DIR}/${name}")
      configure_file("${source}" "${NDP_PYTHON_OUTPUT_DIR}/${name}" COPYONLY)
    endif()
  endforeach()
  if(BUILD_TESTING)
    get_filename_component(NDP_PYTHON_PACKAGE_ROOT "${NDP_PYTHON_OUTPUT_DIR}" DIRECTORY)
    add_test(NAME python_contracts COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${NDP_PYTHON_PACKAGE_ROOT}" "NDP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -v)
    set_tests_properties(python_contracts PROPERTIES TIMEOUT 60)
    if(UNIX AND NDP_GEM5_PIM_WORKER)
      set(NDP_PNM_GEMV_ELF "" CACHE FILEPATH "Optional N3 ELF for PyTorch L1 capture tests")
      set_tests_properties(python_contracts PROPERTIES
        ENVIRONMENT "NDP_GEM5_PIM_WORKER=${NDP_GEM5_PIM_WORKER}")
      set(NDP_PNM_POINTWISE_ELF "" CACHE FILEPATH "Optional N4 pointwise ELF for mixed MLP tests")
      set(NDP_PNM_ATTENTION_ELF "" CACHE FILEPATH "Optional PNM attention ELF for decode tests")
      if(NDP_PNM_GEMV_ELF)
        set_property(TEST python_contracts APPEND PROPERTY ENVIRONMENT
          "NDP_PNM_CAPTURE_WORKER=$<TARGET_FILE:ndp_pnm_test_worker>"
          "NDP_PNM_GEMV_ELF=${NDP_PNM_GEMV_ELF}"
          "NDP_PNM_POINTWISE_ELF=${NDP_PNM_POINTWISE_ELF}"
          "NDP_PNM_ATTENTION_ELF=${NDP_PNM_ATTENTION_ELF}")
      endif()
    endif()
  endif()
endif()
