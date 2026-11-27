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
option(mnet_use_simple_omp "Use mplatform's portable OpenMP runtime instead of system OpenMP" OFF)
option(mnet_enable_dotprod "Enable ARMv8.2 dot-product intrinsics (arm64)" OFF)

if(mnet_use_simple_omp)
    # Must be set before the sibling subprojects are configured; mplatform is
    # added transitively through mformat -> mmat -> mmemory.
    set(MPLATFORM_WITH_SIMPLEOMP ON CACHE BOOL "" FORCE)
else()
    find_package(OpenMP REQUIRED)
endif()

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

# OpenMP abstraction: system OpenMP by default, portable simpleomp when asked.
add_library(mnet_omp INTERFACE)
if(mnet_use_simple_omp)
    # simpleomp is compiled into libmnet.a below, so the public interface only
    # needs the pragma-enabling flag (no external OpenMP runtime to export).
    # Apple clang only exposes OpenMP lowering through the preprocessor flag.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(mnet_omp INTERFACE -Xpreprocessor -fopenmp)
    else()
        target_compile_options(mnet_omp INTERFACE -fopenmp)
    endif()
else()
    target_link_libraries(mnet_omp INTERFACE OpenMP::OpenMP_CXX)
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
    src/core/error.cpp
    src/core/log.cpp
    src/core/layer_registry.cpp
    src/core/net.cpp
    src/gguf/format.cpp
    src/gguf/kernels.cpp
    src/gguf/reader.cpp
    src/tts/exact.cpp
    src/tts/ops.cpp
    src/tts/reference.cpp
    src/tts/tensor.cpp
    src/tokenizer/bpe.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
)

if(mnet_use_simple_omp)
    target_sources(mnet_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../mplatform/src/omp/simpleomp.cpp)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../mplatform/src/omp/simpleomp.cpp
        PROPERTIES COMPILE_OPTIONS "-w")
endif()

target_include_directories(mnet_objs
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(mnet_objs SYSTEM PRIVATE
    $<TARGET_PROPERTY:mlayer::mlayer,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_options(mnet_objs PRIVATE ${MNET_WARNINGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # The quantised kernels spell out every fused multiply-add themselves; keep
    # the compiler from contracting the remaining a*b+c expressions so the
    # library rounds the same on FMA and non-FMA targets.
    target_compile_options(mnet_objs PRIVATE -ffp-contract=off)
endif()
if(mnet_enable_avx2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(mnet_objs PRIVATE -mavx2 -mfma)
endif()
if(mnet_enable_dotprod AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    target_compile_options(mnet_objs PRIVATE -march=armv8.2-a+dotprod)
endif()
target_link_libraries(mnet_objs PRIVATE mformat::mformat mgraph::mgraph mlayer::mlayer mnet_omp)

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
if(mnet_enable_dotprod AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    target_compile_options(mnet INTERFACE -march=armv8.2-a+dotprod)
endif()
target_link_libraries(mnet PUBLIC mnet_omp)
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

    install(TARGETS mnet mnet_omp EXPORT mnetTargets
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

    add_executable(mnet_probe tools/llm/mnet_probe.cpp)
    target_link_libraries(mnet_probe PRIVATE mnet::mnet)
    target_compile_options(mnet_probe PRIVATE ${MNET_WARNINGS})

    add_executable(mnet_gguf tools/model/mnet_gguf.cpp)
    target_link_libraries(mnet_gguf PRIVATE mnet::mnet)
    target_compile_options(mnet_gguf PRIVATE ${MNET_WARNINGS})

    add_executable(mnet_bonsai_block tools/llm/mnet_bonsai_block.cpp)
    target_link_libraries(mnet_bonsai_block PRIVATE mnet::mnet)
    target_compile_options(mnet_bonsai_block PRIVATE ${MNET_WARNINGS})

    # The bit-exactness runners replicate llama.cpp's 8-lane kernels. On x86_64
    # they use the AVX2/FMA intrinsics directly; elsewhere the same lane layout
    # and reduction order is evaluated scalar (mnet/detail/ggml_vec_exact.hpp),
    # so the binaries build and produce the same numbers on arm64 as well.
    add_executable(mnet_bonsai tools/llm/mnet_bonsai.cpp)
    target_link_libraries(mnet_bonsai PRIVATE mnet::mnet)
    target_compile_options(mnet_bonsai PRIVATE ${MNET_WARNINGS})

    add_executable(mnet_inkling2 tools/llm/mnet_inkling2.cpp)
    target_link_libraries(mnet_inkling2 PRIVATE mnet::mnet)
    target_compile_options(mnet_inkling2 PRIVATE ${MNET_WARNINGS})

    add_executable(mnet_inkling2_attn tools/llm/mnet_inkling2_attn.cpp)
    target_link_libraries(mnet_inkling2_attn PRIVATE mnet::mnet)
    target_compile_options(mnet_inkling2_attn PRIVATE ${MNET_WARNINGS})

    # The two exactness runners must produce the same bytes on every target.
    # GCC/Clang default to -ffp-contract=fast, so on x86 the model code fuses
    # a*b+c into FMA while arm64 and the scalar fallbacks do not; the 1 ulp
    # difference then amplifies through MoE routing. The quantised kernels spell
    # out their own fma where llama.cpp fuses, so keeping the model arithmetic
    # unfused is enough to make AVX2, scalar and arm64 agree.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(mnet_bonsai PRIVATE -ffp-contract=off)
        target_compile_options(mnet_inkling2 PRIVATE -ffp-contract=off)
    endif()

    add_executable(mnet_tokenize tools/model/mnet_tokenize.cpp)
    target_link_libraries(mnet_tokenize PRIVATE mnet::mnet)
    target_compile_options(mnet_tokenize PRIVATE ${MNET_WARNINGS})

    add_executable(mnet_tts tools/tts/mnet_tts.cpp tools/tts/mnet_tts_piper.cpp tools/tts/mnet_tts_kokoro.cpp)
    target_link_libraries(mnet_tts PRIVATE mnet::mnet)
    target_include_directories(mnet_tts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    target_compile_options(mnet_tts PRIVATE ${MNET_WARNINGS})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # The Kokoro harmonic STFT reproduces ONNX Runtime's float32 Bluestein
        # DFT; contraction into FMA changes the last bits of its silent-bin
        # phases, so keep the arithmetic unfused.
        target_compile_options(mnet_tts PRIVATE -ffp-contract=off)
    endif()

    add_executable(mnet_llm tools/llm/mnet_llm.cpp)
    target_link_libraries(mnet_llm PRIVATE mnet::mnet)
    target_compile_options(mnet_llm PRIVATE ${MNET_WARNINGS})

    add_executable(mnet_ocr tools/vision/mnet_ocr.cpp)
    target_link_libraries(mnet_ocr PRIVATE mnet::mnet)
    target_compile_options(mnet_ocr PRIVATE ${MNET_WARNINGS})
endif()

if(mnet_build_tests AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
