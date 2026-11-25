cmake_minimum_required(VERSION 3.24)

project(
    mnet
    VERSION 0.1.0
    DESCRIPTION "mnet — C++20 library"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

option(mnet_build_tests "Build mnet tests" ${PROJECT_IS_TOP_LEVEL})
option(mnet_enable_avx2 "Enable AVX2/FMA fast paths (x86_64)" OFF)

find_package(OpenMP REQUIRED)

# Sibling modules are consumed from source when mnet is built inside the mcnn
# tree; installed packages work through the exported target dependencies.
if(NOT TARGET mformat::mformat)
    set(MFORMAT_MMAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mmat" CACHE PATH "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../mformat" "${CMAKE_CURRENT_BINARY_DIR}/mformat")
endif()
if(NOT TARGET mgraph::mgraph)
    set(MGRAPH_MFORMAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mformat" CACHE PATH "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../mgraph" "${CMAKE_CURRENT_BINARY_DIR}/mgraph")
endif()
if(NOT TARGET mlayer::mlayer)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../mlayer" "${CMAKE_CURRENT_BINARY_DIR}/mlayer")
endif()

# 警告基线：per-target，禁止 add_compile_options（会污染第三方子项目）。
set(MNET_WARNINGS_COMMON
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wold-style-cast
    -Wcast-qual
    -Wdouble-promotion
    -Wformat=2
    -Wundef
    -Wpointer-arith
    -Wwrite-strings
    -Wnull-dereference
    -Wzero-as-null-pointer-constant
)

set(MNET_WARNINGS_GCC
    -Wmissing-declarations
    -Wredundant-decls
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast
)

set(MNET_WARNINGS
    ${MNET_WARNINGS_COMMON}
    $<$<CXX_COMPILER_ID:GNU>:${MNET_WARNINGS_GCC}>
)

# Vendored Apache-2.0 BPE tokenizer keeps its upstream warning level.
set_source_files_properties(src/tokenizer/bpe_tokenizer.cpp PROPERTIES COMPILE_OPTIONS "-w")

add_library(mnet_objs OBJECT
    src/mnet/base/error.cpp
    src/mnet/log.cpp
    src/layer_registry.cpp
    src/net.cpp
    src/tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
)

target_include_directories(mnet_objs
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(mnet_objs SYSTEM PRIVATE
    $<TARGET_PROPERTY:mlayer::mlayer,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_options(mnet_objs PRIVATE ${MNET_WARNINGS})
if(mnet_enable_avx2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(mnet_objs PRIVATE -mavx2 -mfma)
endif()
target_link_libraries(mnet_objs PRIVATE mformat::mformat mgraph::mgraph mlayer::mlayer OpenMP::OpenMP_CXX)

add_library(mnet STATIC
    $<TARGET_OBJECTS:mnet_objs>
    $<TARGET_OBJECTS:mgraph_objs>
    $<TARGET_OBJECTS:mformat_objs>
    $<TARGET_OBJECTS:mmat_objs>
    $<TARGET_OBJECTS:mmemory_objs>
    $<TARGET_OBJECTS:mplatform_objs>)
if(mnet_enable_avx2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(mnet INTERFACE -mavx2 -mfma)
endif()
target_link_libraries(mnet PUBLIC OpenMP::OpenMP_CXX)
add_library(mnet::mnet ALIAS mnet)

target_include_directories(mnet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mlayer/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mgraph/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mgraph/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mformat/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mformat/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mmat/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mformat/mmat/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mmemory/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mformat/mmat/mmemory/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mplatform/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mformat/mmat/mmemory/mplatform/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Packaging/export is added once mlayer/mgraph/mformat are installed packages.

if(PROJECT_IS_TOP_LEVEL)
    include(CMakePackageConfigHelpers)

    install(TARGETS mnet EXPORT mnetTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mlayer/include/mlayer
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mgraph/include/mgraph
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mgraph/include/mgraph/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mgraph)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mformat/include/mformat
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mformat/include/mformat/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mformat)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mmat/include/mmat
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mformat/mmat/include/mmat/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mmat)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mmemory/include/mmemory
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mformat/mmat/mmemory/include/mmemory/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mmemory)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mplatform/include/mplatform
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mformat/mmat/mmemory/mplatform/include/mplatform/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mplatform)

    install(EXPORT mnetTargets NAMESPACE mnet:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mnet)
    configure_package_config_file(cmake/mnetConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/mnetConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mnet)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mnetConfigVersion.cmake
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mnetConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/mnetConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mnet)

    add_executable(mnet_probe tools/mnet_probe.cpp)
    target_link_libraries(mnet_probe PRIVATE mnet::mnet)
    target_compile_options(mnet_probe PRIVATE ${MNET_WARNINGS})
endif()

if(mnet_build_tests AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
