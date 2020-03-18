cmake_minimum_required(VERSION 3.2)
project(cvm C CXX)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindCUDA.cmake)
include(cmake/util/FindOpenmp.cmake)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
cvm_option(USE_CUDA "Build with CUDA" ON)
cvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" ON)
cvm_option(USE_CVM_RUNTIME "Build with tiny cvm runtime" ON)
cvm_option(USE_RTTI "Build with RTTI" ON)
cvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)

# include directories
include_directories("include")

# initial variables
set(CVM_LINKER_LIBS "")
set(CVM_RUNTIME_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generic compilation options
include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  if ("${DEBUG}" STREQUAL "ON")
  set(CMAKE_C_FLAGS "-fsigned-char -Og -g -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fsigned-char -Og -g -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_C_FLAGS "-fsigned-char -O3 -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fsigned-char -O3 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "-mavx2 -O2 -fno-strict-aliasing -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")}"
endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

# add source group
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB RUNTIME_SRCS src/*.cc)

if(${PROFILE} STREQUAL "ON")
    add_definitions(-DPROFILE)
endif()

message(STATUS "Build with CVM runtime support...")
file(GLOB RUNTIME_CVM_SRCS src/cvm/*.cc) 
if(${USE_CUDA} STREQUAL "ON")
    message("use cuda")
    project(cvm CUDA)
    set (EXECUTE_NAME cvm_runtime_cuda)
    add_definitions(-DCVM_RUNTIME_CUDA)
    add_definitions(-DCUDA_PROFILE)
    file(GLOB RUNTIME_CVM_CUDA_SRCS src/cuda/*.cc)
    file(GLOB CVM_OPS_CUDA_SRCS src/cvm/ops/cuda/*.cc
      src/cvm/ops/cuda/*.cu)
    list(APPEND RUNTIME_CVM_SRCS ${CVM_OPS_CUDA_SRCS})
    list(APPEND RUNTIME_SRCS ${RUNTIME_CVM_CUDA_SRCS})
    list(APPEND CVM_RUNTIME_LINKER_LIBS "cudart")
    list(APPEND CVM_RUNTIME_LINKER_LIBS "cuda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 --expt-extended-lambda")
elseif(${USE_FORMAL} STREQUAL "ON")
    message("use formal")
    file(GLOB CVM_OPS_CPU_SRCS src/cvm/ops/formal/*.cc)
    list(APPEND RUNTIME_CVM_SRCS ${CVM_OPS_CPU_SRCS})
    set (EXECUTE_NAME cvm_runtime_formal)
else()
    file(GLOB CVM_OPS_CPU_SRCS src/cvm/ops/cpu/*.cc)
    list(APPEND RUNTIME_CVM_SRCS ${CVM_OPS_CPU_SRCS})
    set (EXECUTE_NAME cvm_runtime_cpu)
endif()
list(APPEND RUNTIME_SRCS ${RUNTIME_CVM_SRCS})
file(GLOB CVM_TOP_SRCS src/cvm/top/*.cc
          src/cvm/top/nn/*.cc
          src/cvm/top/tensor/*.cc)
list(APPEND RUNTIME_SRCS ${CVM_TOP_SRCS})

# Module rules
include(cmake/modules/CUDA.cmake)

add_library(${EXECUTE_NAME} SHARED ${RUNTIME_SRCS})
target_link_libraries(${EXECUTE_NAME} ${CVM_RUNTIME_LINKER_LIBS})

# Installation rules
install(TARGETS ${EXECUTE_NAME} DESTINATION lib${LIB_SUFFIX})

# More target definitions
if(MSVC)
    target_compile_definitions(${EXECUTE_NAME} PRIVATE -DHalide_EXPORTS)
    target_compile_definitions(${EXECUTE_NAME} PRIVATE -DCVM_EXPORTS)
endif()
