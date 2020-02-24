ROOTDIR = $(CURDIR)

.PHONY: clean all test

INCLUDE_FLAGS = -Iinclude
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =

all: cpu gpu formal 

cpu:
		 @mkdir -p build/cpu && cd build/cpu && cmake ../.. -DUSE_CUDA=OFF && $(MAKE)
#		@mkdir -p build/cpu && cd build/cpu && cmake ../.. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug && $(MAKE)

gpu:
		 @mkdir -p build/gpu && cd build/gpu && cmake ../.. -DUSE_CUDA=ON && $(MAKE)
# @mkdir -p build/gpu && cd build/gpu && cmake ../.. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug && $(MAKE)
formal:
		 @mkdir -p build/formal && cd build/formal && cmake ../.. -DUSE_CUDA=OFF -DUSE_FORMAL=ON && $(MAKE)
clean:
	  rm -rf ./build/*
