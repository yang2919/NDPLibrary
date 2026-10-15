# Build the NDP library, runnable examples, and contract test suites.
cmake_minimum_required(VERSION 3.20)
project(ndp_library LANGUAGES CXX)

find_package(Threads REQUIRED)

add_library(ndp
  src/common/compute.cpp
  src/common/fp16.cpp
  src/common/tensor_metadata.cpp
  src/l1/functional_backend.cpp
  src/l1/functional_ops.cpp
  src/l2/session.cpp
)
target_include_directories(ndp PUBLIC include)
target_compile_features(ndp PUBLIC cxx_std_20)
target_link_libraries(ndp PRIVATE Threads::Threads)
target_compile_options(ndp PRIVATE -Wall -Wextra -Wpedantic)

# L3 depends on L2; the core library has no cache/model dependency.
add_library(ndp_llm src/l3/llm.cpp src/l3/decoder.cpp)
target_link_libraries(ndp_llm PUBLIC ndp)
target_compile_options(ndp_llm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(single_device_linear examples/single_device_linear.cpp)
target_link_libraries(single_device_linear PRIVATE ndp)

add_executable(decoder_block examples/decoder_block.cpp)
target_link_libraries(decoder_block PRIVATE ndp_llm)

# Examples remain available when contract tests are disabled.
include(CTest)
if(BUILD_TESTING)
  foreach(name session async layout compute buffer)
    add_executable(ndp_${name}_tests tests/${name}_test.cpp)
    target_link_libraries(ndp_${name}_tests PRIVATE ndp)
    add_test(NAME ${name}_contracts COMMAND ndp_${name}_tests)
    set_tests_properties(${name}_contracts PROPERTIES TIMEOUT 20)
  endforeach()

  foreach(name llm cache decoder)
    add_executable(ndp_${name}_tests tests/${name}_test.cpp)
    target_link_libraries(ndp_${name}_tests PRIVATE ndp_llm)
    add_test(NAME ${name}_contracts COMMAND ndp_${name}_tests)
    set_tests_properties(${name}_contracts PROPERTIES TIMEOUT 20)
  endforeach()
endif()
