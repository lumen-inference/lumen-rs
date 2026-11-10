cmake_minimum_required(VERSION 3.21)
project(velri VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
include(CTest)

# Consume the separate computation library, never copies of its private kernels.
set(VELRI_LIB_DEFAULT_SOURCE "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/velri-lib/CMakeLists.txt")
    set(VELRI_LIB_DEFAULT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/external/velri-lib")
endif()
set(VELRI_LIB_SOURCE_DIR "${VELRI_LIB_DEFAULT_SOURCE}" CACHE PATH "Path to a separate Velri-lib source checkout")
if(VELRI_LIB_SOURCE_DIR)
    get_filename_component(VELRI_LIB_SOURCE_DIR "${VELRI_LIB_SOURCE_DIR}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${VELRI_LIB_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "VELRI_LIB_SOURCE_DIR must point to a Velri-lib checkout")
    endif()
    set(VELRI_LIB_BUILD_TESTS ${BUILD_TESTING} CACHE BOOL "Build dependency correctness tests")
    add_subdirectory("${VELRI_LIB_SOURCE_DIR}" velri-lib)
else()
    find_package(VelriLib 0.1.0 EXACT CONFIG REQUIRED)
endif()

# Only an integration check exists; the engine runtime and graph are not ready.
if(BUILD_TESTING)
    add_executable(test_velri_lib tests/integration/velri_lib.c)
    target_link_libraries(test_velri_lib PRIVATE VelriLib::cpu_linear)
    add_test(NAME velri_lib_integration COMMAND test_velri_lib)
endif()
