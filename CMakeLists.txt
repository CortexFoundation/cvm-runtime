cmake_minimum_required(VERSION 3.2)
project(cvm C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON )

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
# cvm_option(USE_CUDA "Build with CUDA" ON)

# include directories
include_directories("include")

# initial variables
set(CVM_RUNTIME_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generic compilation options
set(CMAKE_C_FLAGS "-fsigned-char -Wall -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fsigned-char -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
if (DEBUG)
  set(CMAKE_C_FLAGS "-Og -g ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Og -g ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_C_FLAGS "-O3 ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
endif ()

# Source file lists
file(GLOB RUNTIME_SRCS src/core/*.cc
                       src/runtime/*.cc
                       src/runtime/cpu/*.cc
                       src/runtime/formal/*.cc)

if(PROFILE)
  add_definitions(-DPROFILE)
  if (USE_CUDA)
    add_definitions(-DCUDA_PROFILE)
  endif()
endif()

message(STATUS "Build with CVM runtime support...")
file(GLOB CVM_SRCS src/cvm/*.cc src/cvm/compiler/*.cc src/cvm/pass/*.cc) 
if(USE_CUDA)
    message("USE CUDA")
    # project(cvm CUDA)

    find_cuda(${USE_CUDA})
    if (CUDA_FOUND)
      include_directories(${CUDA_INCLUDE_DIRS})
    else()
      message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
    endif()

    list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND CVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

    # set(CMAKE_CUDA_FLAGS
    #     "${CMAKE_CUDA_FLAGS} -std=c++11 --expt-extended-lambda")
    set(CUDA_NVCC_FLAGS 
        ${CUDA_NVCC_FLAGS} -std=c++11 --expt-extended-lambda)

    set (EXECUTE_NAME cvm_runtime_cuda)
    add_definitions(-DCVM_RUNTIME_CUDA)
      
    message(STATUS "Build with CUDA support")
    file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})

    file(GLOB CVM_OPS_CUDA_SRCS src/cvm/ops/cuda/*.cc)
    list(APPEND CVM_SRCS ${CVM_OPS_CUDA_SRCS})

    file(GLOB CUDA_SRCS src/cvm/ops/cuda/*.cu)
    cuda_compile(CUDA_LIBS ${CUDA_SRCS})
    list(APPEND CVM_SRCS ${CUDA_LIBS})
elseif(USE_FORMAL)
    message("USE FORMAL")
    file(GLOB CVM_OPS_CPU_SRCS src/cvm/ops/formal/*.cc)
    list(APPEND CVM_SRCS ${CVM_OPS_CPU_SRCS})
    set (EXECUTE_NAME cvm_runtime_formal)
else()
    message("USE CPU")
    file(GLOB CVM_OPS_CPU_SRCS src/cvm/ops/cpu/*.cc)
    list(APPEND CVM_SRCS ${CVM_OPS_CPU_SRCS})
    set (EXECUTE_NAME cvm_runtime_cpu)
endif()
list(APPEND RUNTIME_SRCS ${CVM_SRCS})

file(GLOB CVM_TOP_SRCS src/top/*.cc
                       src/top/nn/*.cc
                       src/top/tensor/*.cc)
list(APPEND RUNTIME_SRCS ${CVM_TOP_SRCS})

add_library(${EXECUTE_NAME} SHARED ${RUNTIME_SRCS})
target_link_libraries(${EXECUTE_NAME} ${CVM_RUNTIME_LINKER_LIBS})
