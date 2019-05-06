cmake_minimum_required(VERSION 3.2)
project(cvm C CXX)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindCUDA.cmake)
include(cmake/util/FindOpenmp.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
cvm_option(USE_CUDA "Build with CUDA" ON)
cvm_option(USE_OPENCL "Build with OpenCL" ON)
cvm_option(USE_VULKAN "Build with Vulkan" OFF)
cvm_option(USE_METAL "Build with Metal" OFF)
cvm_option(USE_ROCM "Build with ROCM" OFF)
cvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
cvm_option(USE_RPC "Build with RPC" ON)
cvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" ON)
cvm_option(USE_STACKVM_RUNTIME "Include stackvm into the runtime" OFF)
cvm_option(USE_GRAPH_RUNTIME "Build with tiny graph runtime" ON)
cvm_option(USE_CVM_RUNTIME "Build with tiny cvm runtime" ON)
cvm_option(USE_GRAPH_RUNTIME_DEBUG "Build with tiny graph runtime debug mode" OFF)
cvm_option(USE_CVM_GRAPH_RUNTIME "Build with CVM graph runtime" ON)
cvm_option(USE_SGX "Build with SGX" OFF)
cvm_option(USE_RTTI "Build with RTTI" ON)
cvm_option(USE_MSVC_MT "Build with MT" OFF)
cvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)

# Contrib library options
cvm_option(USE_BLAS "The blas library to be linked" none)
cvm_option(USE_MKL_PATH "MKL root path when use MKL blas" none)
cvm_option(USE_CUDNN "Build with cuDNN" OFF)
cvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
cvm_option(USE_MIOPEN "Build with ROCM:MIOpen" OFF)
cvm_option(USE_ROCBLAS "Build with ROCM:RoCBLAS" OFF)
cvm_option(USE_SORT "Build with sort support" OFF)
cvm_option(USE_NNPACK "Build with nnpack support" OFF)
cvm_option(USE_RANDOM "Build with random support" OFF)
cvm_option(USE_ANTLR "Build with ANTLR for Relay parsing" OFF)

# include directories
include_directories("include")

# initial variables
set(CVM_LINKER_LIBS "")
set(CVM_RUNTIME_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_definitions(-DHalide_SHARED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  message(STATUS "BUILD C FLAGS")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_compile_options(-O0 -Wall -fPIC -fvisibility=hidden -std=c++11)
  else()
    set(CMAKE_C_FLAGS "-g -O2 -Wall -fPIC -fvisibility=hidden ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-g -O2 -Wall -fPIC -fvisibility=hidden -std=c++11 ${CMAKE_CXX_FLAGS}")
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()
endif(MSVC)

# add source group
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB RUNTIME_SRCS src/runtime/*.cc)

# Package runtime rules
if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  file(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

file(GLOB STACKVM_RUNTIME_SRCS src/runtime/stackvm/*.cc)
list(APPEND COMPILER_SRCS ${STACKVM_CODEGEN_SRCS})
if(USE_STACKVM_RUNTIME)
  message(STATUS "Build with stackvm support in runtime...")
  list(APPEND RUNTIME_SRCS ${STACKVM_RUNTIME_SRCS})
else()
  list(APPEND COMPILER_SRCS ${STACKVM_RUNTIME_SRCS})
endif(USE_STACKVM_RUNTIME)

message(STATUS "Build with CVM runtime support...")
file(GLOB RUNTIME_CVM_SRCS src/runtime/cvm/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_CVM_SRCS})
file(GLOB CVM_TOP_SRCS src/runtime/cvm/top/*.cc
          src/runtime/cvm/top/nn/*.cc
          src/runtime/cvm/top/tensor/*.cc)
list(APPEND RUNTIME_SRCS ${CVM_TOP_SRCS})

if(USE_GRAPH_RUNTIME)
  message(STATUS "Build with Graph runtime support...")
  file(GLOB RUNTIME_GRAPH_SRCS src/runtime/graph/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_SRCS})

  if(USE_GRAPH_RUNTIME_DEBUG)
    message(STATUS "Build with Graph runtime debug support...")
    file(GLOB RUNTIME_GRAPH_DEBUG_SRCS src/runtime/graph/debug/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_DEBUG_SRCS})
    set_source_files_properties(${RUNTIME_GRAPH_SRCS}
      PROPERTIES COMPILE_DEFINITIONS "CVM_GRAPH_RUNTIME_DEBUG")
  endif(USE_GRAPH_RUNTIME_DEBUG)
endif(USE_GRAPH_RUNTIME)


# Module rules
# include(cmake/modules/VTA.cmake)
include(cmake/modules/CUDA.cmake)

add_library(cvm_runtime SHARED ${RUNTIME_SRCS})
target_link_libraries(cvm_runtime ${CVM_RUNTIME_LINKER_LIBS})

# Tests
# Custom targets
add_custom_target(runtime DEPENDS cvm_runtime)

# Installation rules
install(TARGETS cvm_runtime DESTINATION lib${LIB_SUFFIX})

if (INSTALL_DEV)
  install(
    DIRECTORY "include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
else(INSTALL_DEV)
  install(
    DIRECTORY "include/cvm/runtime/." DESTINATION "include/cvm/runtime"
    FILES_MATCHING
    PATTERN "*.h"
    )
endif(INSTALL_DEV)

# More target definitions
if(MSVC)
  target_compile_definitions(cvm_runtime PRIVATE -DHalide_EXPORTS)
  target_compile_definitions(cvm_runtime PRIVATE -DCVM_EXPORTS)
endif()
