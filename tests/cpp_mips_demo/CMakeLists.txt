cmake_minimum_required(VERSION 3.2)
project(cvm C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER "/usr/bin/mips-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/mips-linux-gnu-g++")
#add_compile_options(-Wall -Werror -static -mips32r5 -O3)
add_compile_options(-Wall -Werror -static -mfp32 -march=R2000)
# 指定目标系统的类型
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips)
set(CMAKE_EXE_LINKER_FLAGS "-static -mfp32 -march=R2000")

# Utility functions
include(conf.cmake)

include_directories("include")

#file(GLOB CVM_TOP_SRCS src/*.cc)

#add_library(demo SHARED src/main.cc)
#target_link_libraries(demo atomic)

add_executable(main src/main.cc)
target_compile_options(main
    PUBLIC -Wall -static -mfp32 -march=R2000)
target_link_libraries(main atomic)
