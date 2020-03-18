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

# include directories
include_directories("include")

# initial variables
set(CVM_LINKER_LIBS "")
set(CVM_RUNTIME_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generic compilation options
set(CMAKE_C_FLAGS "-fsigned-char -Wall -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fsigned-char -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
if ("${DEBUG}" STREQUAL "ON")
  set(CMAKE_C_FLAGS "-Og -g ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Og -g ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_C_FLAGS "-O3 ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
endif ()

# Source file lists
file(GLOB RUNTIME_SRCS src/*.cc)

if(${PROFILE} STREQUAL "ON")
    add_definitions(-DPROFILE)
endif()

message(STATUS "Build with CVM runtime support...")
file(GLOB RUNTIME_CVM_SRCS src/cvm/*.cc) 
if(${USE_CUDA} STREQUAL "ON")
    message("USE CUDA")
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
    message("USE FORMAL")
    file(GLOB CVM_OPS_CPU_SRCS src/cvm/ops/formal/*.cc)
    list(APPEND RUNTIME_CVM_SRCS ${CVM_OPS_CPU_SRCS})
    set (EXECUTE_NAME cvm_runtime_formal)
else()
    message("USE CPU")
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
