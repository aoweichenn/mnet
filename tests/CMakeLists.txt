find_package(GTest CONFIG REQUIRED)

add_library(mnet_test_main STATIC support/gtest_main.cpp)
target_link_libraries(mnet_test_main PUBLIC GTest::gtest)
target_compile_options(mnet_test_main PRIVATE ${MNET_WARNINGS_COMMON})

function(mnet_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mnet::mnet mnet_test_main)
    target_include_directories(${name} SYSTEM PRIVATE
        $<TARGET_PROPERTY:mlayer::mlayer,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_options(${name} PRIVATE ${MNET_WARNINGS_COMMON})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

mnet_add_test(mnet_result_tests unit/base/result_tests.cpp)
mnet_add_test(mnet_error_tests unit/base/error_tests.cpp)
mnet_add_test(mnet_log_tests unit/base/log_tests.cpp)
mnet_add_test(mnet_memory_budget_tests unit/base/memory_budget_tests.cpp)
mnet_add_test(mnet_net_tests unit/net_tests.cpp)
mnet_add_test(mnet_model_tests unit/model_tests.cpp)
mnet_add_test(mnet_kernel_tests unit/kernel_tests.cpp)
