find_package(GTest CONFIG REQUIRED)

add_library(mnet_test_main STATIC support/gtest_main.cpp)
target_link_libraries(mnet_test_main PUBLIC GTest::gtest)
target_compile_options(mnet_test_main PRIVATE ${MNET_WARNINGS_COMMON})

function(mnet_test_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE mnet::mnet mnet_test_main)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/support)
    target_include_directories(${name} SYSTEM PRIVATE
        $<TARGET_PROPERTY:mlayer::mlayer,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_options(${name} PRIVATE ${MNET_WARNINGS_COMMON})
endfunction()

function(mnet_add_test name)
    mnet_test_executable(${name} ${ARGN})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

mnet_add_test(mnet_result_tests unit/base/result_tests.cpp)
mnet_add_test(mnet_error_tests unit/base/error_tests.cpp)
mnet_add_test(mnet_log_tests unit/base/log_tests.cpp)
mnet_add_test(mnet_memory_budget_tests unit/base/memory_budget_tests.cpp)
mnet_add_test(mnet_net_tests unit/net_tests.cpp)
mnet_add_test(mnet_model_tests unit/model_tests.cpp)
mnet_add_test(mnet_model_bin_tests unit/format/model_bin_tests.cpp)
set_tests_properties(mnet_model_bin_tests PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED")

include(GoogleTest)
function(mnet_add_operator_tests group)
    set(name mnet_operators_${group})
    mnet_test_executable(${name} ${ARGN})
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST TEST_PREFIX "operators.${group}."
        PROPERTIES LABELS "operators;operators.${group}")
endfunction()

add_subdirectory(unit/operators)

# Timings are for manual investigation, not correctness gates.
mnet_test_executable(mnet_operator_benchmarks ../benchmarks/operators_bench.cpp)
set_target_properties(mnet_operator_benchmarks PROPERTIES EXCLUDE_FROM_ALL TRUE)
