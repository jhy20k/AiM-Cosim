# =============================================================================
# AiM-Cosim - AiM Simulator + Verilator Co-verification Framework
# Top-level CMakeLists.txt
# =============================================================================

cmake_minimum_required(VERSION 3.15)

project(
  AiM_Cosim
  VERSION 1.0
  LANGUAGES CXX
)

#### Build Configuration ####
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "Configuring ${CMAKE_PROJECT_NAME} v${CMAKE_PROJECT_VERSION}...")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

if(DEFINED AIMVSIM_ENABLE_LTO AND NOT DEFINED AIM_COSIM_ENABLE_LTO)
    set(AIM_COSIM_ENABLE_LTO "${AIMVSIM_ENABLE_LTO}" CACHE BOOL "Enable interprocedural optimization for Release builds" FORCE)
endif()
if(DEFINED AIMVSIM_ENABLE_VCD_TRACE AND NOT DEFINED AIM_COSIM_ENABLE_VCD_TRACE)
    set(AIM_COSIM_ENABLE_VCD_TRACE "${AIMVSIM_ENABLE_VCD_TRACE}" CACHE BOOL "Build Verilator model with VCD tracing support" FORCE)
endif()
if(DEFINED AIMVSIM_NATIVE_OPT AND NOT DEFINED AIM_COSIM_NATIVE_OPT)
    set(AIM_COSIM_NATIVE_OPT "${AIMVSIM_NATIVE_OPT}" CACHE BOOL "Enable host-native code generation for local benchmarking" FORCE)
endif()
if(DEFINED AIMVSIM_VERILATOR_THREADS AND NOT DEFINED AIM_COSIM_VERILATOR_THREADS)
    set(AIM_COSIM_VERILATOR_THREADS "${AIMVSIM_VERILATOR_THREADS}" CACHE STRING "Verilator model threads. Use >1 only after benchmarking this RTL on the target host." FORCE)
endif()

option(AIM_COSIM_ENABLE_LTO "Enable interprocedural optimization for Release builds" ON)
option(AIM_COSIM_ENABLE_VCD_TRACE "Build Verilator model with VCD tracing support" OFF)
option(AIM_COSIM_NATIVE_OPT "Enable host-native code generation for local benchmarking" OFF)
set(AIM_COSIM_VERILATOR_THREADS "1" CACHE STRING "Verilator model threads. Use >1 only after benchmarking this RTL on the target host.")
if(NOT AIM_COSIM_VERILATOR_THREADS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "AIM_COSIM_VERILATOR_THREADS must be a positive integer")
endif()

if(AIM_COSIM_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT aim_cosim_ipo_supported OUTPUT aim_cosim_ipo_error)
    if(aim_cosim_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "Link-Time Optimization (LTO) enabled")
    else()
        message(WARNING "LTO requested but not supported: ${aim_cosim_ipo_error}")
    endif()
endif()

if(AIM_COSIM_NATIVE_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<CONFIG:Release>:-march=native>)
    message(STATUS "Native CPU optimization enabled (-march=native)")
endif()

# =============================================================================
# 1. Verilator
# =============================================================================
find_package(verilator REQUIRED HINTS $ENV{VERILATOR_ROOT})
message(STATUS "Verilator found: ${VERILATOR_ROOT}")

# =============================================================================
# 2. aim_simulator (produces `ramulator` shared lib + `ramulator-exe`)
#    Requires patched CMakeLists.txt (CMAKE_SOURCE_DIR → CMAKE_CURRENT_SOURCE_DIR)
# =============================================================================
add_subdirectory(extern/aim_simulator)

# =============================================================================
# 3. Verilate RTL → static library
# =============================================================================
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(RTL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/rtl)
set(ACC_MAG_WIDTH "32" CACHE STRING "Accumulator magnitude width (32, 37, 50)")

if(ACC_MAG_WIDTH STREQUAL "32")
    set(RTL_TOP pim_mac_tree_mag32)
    set(RTL_PREFIX Vpim_mac_tree_mag32)
    set(RTL_TOP_FILE ${RTL_DIR}/pim_mac_tree_mag32.v)
    set(RTL_ACC_FILE ${RTL_DIR}/accumulator_param.v)
elseif(ACC_MAG_WIDTH STREQUAL "37")
    set(RTL_TOP pim_mac_tree_mag37)
    set(RTL_PREFIX Vpim_mac_tree_mag37)
    set(RTL_TOP_FILE ${RTL_DIR}/pim_mac_tree_mag37.v)
    set(RTL_ACC_FILE ${RTL_DIR}/accumulator_param.v)
else()
    set(RTL_TOP pim_mac_tree)
    set(RTL_PREFIX Vpim_mac_tree)
    set(RTL_TOP_FILE ${RTL_DIR}/pim_mac_tree.v)
    set(RTL_ACC_FILE ${RTL_DIR}/accumulator.v)
endif()

add_library(verilated_model STATIC)
set(AIM_COSIM_VERILATOR_ARGS
    --cc
    -Wall -Wno-fatal
    --x-assign fast
    --x-initial fast
    --no-assert
    -O3
)

if(AIM_COSIM_ENABLE_VCD_TRACE)
    list(APPEND AIM_COSIM_VERILATOR_ARGS --trace)
    message(STATUS "Verilator VCD tracing support: enabled")
else()
    message(STATUS "Verilator VCD tracing support: disabled")
endif()

if(NOT AIM_COSIM_VERILATOR_THREADS STREQUAL "1")
    list(APPEND AIM_COSIM_VERILATOR_ARGS --threads ${AIM_COSIM_VERILATOR_THREADS})
    find_package(Threads REQUIRED)
    message(STATUS "Verilator model threads: ${AIM_COSIM_VERILATOR_THREADS}")
else()
    message(STATUS "Verilator model threads: 1")
endif()

verilate(verilated_model
    SOURCES
        ${RTL_TOP_FILE}
        ${RTL_ACC_FILE}
        ${RTL_DIR}/activation_unit.v
        ${RTL_DIR}/adder_tree.v
        ${RTL_DIR}/bf16_to_q816.v
        ${RTL_DIR}/pu.v
        ${RTL_DIR}/q816_to_bf16_rne.v
    TOP_MODULE ${RTL_TOP}
    PREFIX ${RTL_PREFIX}
    VERILATOR_ARGS ${AIM_COSIM_VERILATOR_ARGS}
)
message(STATUS "RTL verilated: ${RTL_TOP} (MAG_WIDTH=${ACC_MAG_WIDTH})")
target_compile_definitions(verilated_model PUBLIC
    AIM_COSIM_ENABLE_VCD_TRACE=$<BOOL:${AIM_COSIM_ENABLE_VCD_TRACE}>
)
if(NOT AIM_COSIM_VERILATOR_THREADS STREQUAL "1")
    target_link_libraries(verilated_model PUBLIC Threads::Threads)
endif()

# =============================================================================
# 4. AiM-Cosim bridge library (VerilatorEngine + ComputeEngine + MemoryManager + Bridge + DataLoader)
# =============================================================================
add_library(aim_cosim_bridge STATIC
    src/core/verilator_engine/verilator_engine.cpp
    src/core/compute_engine/aim_compute_engine.cpp
    src/core/compute_engine/hbmpim_compute_engine.cpp
    src/core/hbmpim/pim_cmd.cpp
    src/core/hbmpim/pim_block_model.cpp
    src/core/memory_manager/memory_manager.cpp
    src/core/memory_manager/hbmpim_memory_manager.cpp
    src/core/data_loader/data_loader.cpp
)

target_include_directories(aim_cosim_bridge PUBLIC
    ${CORE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/ext/spdlog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/ext/yaml-cpp/include
    ${VERILATOR_ROOT}/include
    ${VERILATOR_ROOT}/include/vltstd
)

target_compile_definitions(aim_cosim_bridge PUBLIC
    ACC_MAG_WIDTH=${ACC_MAG_WIDTH}
    AIM_COSIM_ENABLE_VCD_TRACE=$<BOOL:${AIM_COSIM_ENABLE_VCD_TRACE}>
)
target_link_libraries(aim_cosim_bridge PUBLIC verilated_model)

# =============================================================================
# 5. Inject AiMRTL controller into ramulator target
#    The RAMULATOR_REGISTER_IMPLEMENTATION macro requires the controller
#    to be linked into the same shared library for factory discovery.
# =============================================================================
target_sources(ramulator PRIVATE
    ${CORE_DIR}/controller/AiMRTL_dram_controller.cpp
    ${CORE_DIR}/controller/divergence_tracker.cpp
    ${CORE_DIR}/controller/result_logger.cpp
    ${CORE_DIR}/controller/functional_dispatcher.cpp
    ${CORE_DIR}/architecture/hbmpim_architecture_backend.cpp
)

target_include_directories(ramulator PRIVATE
    ${CORE_DIR}
    ${VERILATOR_ROOT}/include
    ${VERILATOR_ROOT}/include/vltstd
)

target_link_libraries(ramulator PRIVATE aim_cosim_bridge)

message(STATUS "AiMRTL controller injected into ramulator target")

# =============================================================================
# 6. Test executables
# =============================================================================
add_executable(test_mode2_vs_mode3
    tests/test_mode2_vs_mode3.cpp
)
target_link_libraries(test_mode2_vs_mode3 PRIVATE aim_cosim_bridge)

add_executable(test_mac_latency_probe
    tests/test_mac_latency_probe.cpp
)
target_link_libraries(test_mac_latency_probe PRIVATE aim_cosim_bridge)

add_executable(test_data_loader
    tests/test_data_loader.cpp
)
target_link_libraries(test_data_loader PRIVATE aim_cosim_bridge)

add_executable(test_aim_compute_scoreboard
    tests/test_aim_compute_scoreboard.cpp
)
target_include_directories(test_aim_compute_scoreboard PRIVATE
    ${CORE_DIR}
)
target_link_libraries(test_aim_compute_scoreboard PRIVATE aim_cosim_bridge)

add_executable(test_aim_row_locality
    tests/test_aim_row_locality.cpp
)
target_include_directories(test_aim_row_locality PRIVATE
    ${CORE_DIR}
)

add_executable(test_aim_addr_translate
    tests/test_aim_addr_translate.cpp
)
target_include_directories(test_aim_addr_translate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
)
target_link_libraries(test_aim_addr_translate PRIVATE ramulator)

add_executable(test_aim_region_registry
    tests/test_aim_region_registry.cpp
)
target_include_directories(test_aim_region_registry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
)
target_link_libraries(test_aim_region_registry PRIVATE ramulator)

add_executable(test_aim_trace_parser
    tests/test_aim_trace_parser.cpp
)
target_include_directories(test_aim_trace_parser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
)
target_link_libraries(test_aim_trace_parser PRIVATE ramulator)

add_executable(test_hbmpim_stage5a
    tests/test_hbmpim_stage5a.cpp
)
target_include_directories(test_hbmpim_stage5a PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
)
target_link_libraries(test_hbmpim_stage5a PRIVATE ramulator)

add_executable(test_hbmpim_stage5c
    tests/test_hbmpim_stage5c.cpp
)
target_include_directories(test_hbmpim_stage5c PRIVATE
    ${CORE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/ext/spdlog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/ext/yaml-cpp/include
    ${VERILATOR_ROOT}/include
    ${VERILATOR_ROOT}/include/vltstd
)
target_link_libraries(test_hbmpim_stage5c PRIVATE aim_cosim_bridge ramulator)

add_executable(test_hbmpim_stage5d
    tests/test_hbmpim_stage5d.cpp
)
target_include_directories(test_hbmpim_stage5d PRIVATE
    ${CORE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/ext/spdlog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/aim_simulator/ext/yaml-cpp/include
    ${VERILATOR_ROOT}/include
    ${VERILATOR_ROOT}/include/vltstd
)
target_link_libraries(test_hbmpim_stage5d PRIVATE aim_cosim_bridge ramulator)

message(STATUS "Configuration complete. Build with: cmake --build . -j$(nproc)")
