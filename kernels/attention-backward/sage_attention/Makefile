# Makefile for SAGE Attention HipKittens Kernel
#
# Usage:
#   make                       # build the kernel as a pybind11 module
#   make GPU_TARGET=CDNA3      # target MI300X (gfx942)
#   make GPU_TARGET=CDNA4      # target MI350X (gfx950) [default]
#   make clean                 # remove build artifacts
#
# Parameters can be overridden:
#   make ATTN_B=8 ATTN_H=32 ATTN_H_KV=8 ATTN_N=2048

# Compiler and GPU target
GPU_TARGET ?= CDNA4

# Attention parameters (defaults)
ATTN_B    ?= 4
ATTN_H    ?= 32
ATTN_H_KV ?= 8
ATTN_N    ?= 1024
ATTN_D    ?= 128

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)
CPPFLAGS += -DATTN_D=$(ATTN_D)
CPPFLAGS += -DUSE_PYBIND

# ROCm / HIP paths
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# GPU target flags
ifeq ($(GPU_TARGET),CDNA2)
  HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
  HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
  HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# C++ standard and common flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

# Python / pybind11 flags
PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
             $(shell python3 -m pybind11 --includes) -shared -fPIC \
             -Rpass-analysis=kernel-resource-usage

# Log directory
LOGDIR  := /workdir/data_logs/$(shell date +%m%d_%H%M%S)_outputs
LOGFILE := $(LOGDIR)/make_build.log

# Default target
all: sage_attn_kernel

# Build rule
sage_attn_kernel: kernel.cpp
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $@$(shell python3-config --extension-suffix) 2>&1 | tee $(LOGFILE)

# Clean
clean:
	rm -f sage_attn_kernel*

.PHONY: all clean
