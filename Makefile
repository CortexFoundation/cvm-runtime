ROOTDIR = $(CURDIR)

.PHONY: clean all test runtime

ifndef DMLC_CORE_PATH
  DMLC_CORE_PATH = $(ROOTDIR)/3rdparty/dmlc-core
endif

ifndef DLPACK_PATH
  DLPACK_PATH = $(ROOTDIR)/3rdparty/dlpack
endif

INCLUDE_FLAGS = -Iinclude -I$(DLPACK_PATH)/include -I$(DMLC_CORE_PATH)/include
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

runtime:
	@mkdir -p build && cd build && cmake .. && $(MAKE) runtime

cpptest:
	@mkdir -p build && cd build && cmake .. && $(MAKE) cpptest

clean:
	@mkdir -p build && cd build && cmake .. && $(MAKE) clean
