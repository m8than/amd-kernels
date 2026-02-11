# Makefile for MHA One-Kernel Backward Pass (HipKittens)
#
# Usage:
#   make                           # builds the kernel with default parameters
#   make ATTN_B=8 ATTN_N=2048     # builds with custom batch/seq length
#   make GPU_TARGET=CDNA3          # builds for MI300X (gfx942)
#   make clean                     # removes built artifacts
#
# Produces: tk_mha_onekernel_bwd.cpython-*.so (Python extension module)

# GPU target: CDNA2 (MI250), CDNA3 (MI300X), CDNA4 (MI350X)
GPU_TARGET ?= CDNA4

# Attention parameters (overridable from command line)
ATTN_B    ?= 4
ATTN_H    ?= 32
ATTN_H_KV ?= 8
ATTN_N    ?= 1024

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

# ROCm / HIP compiler
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX           ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Architecture flags
ifeq ($(GPU_TARGET),CDNA2)
  HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
  HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
  HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# C++ standard and include paths
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD) -w
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)

# Python / pybind11 flags
PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
PY_INCLUDES := $(shell python3 -m pybind11 --includes)
PY_EXT_SUFFIX := $(shell python3-config --extension-suffix)

ICXXFLAGS += $(PY_LDFLAGS) $(PY_INCLUDES) -shared -fPIC
ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
ICXXFLAGS += -Rpass-analysis=kernel-resource-usage
ICPPFLAGS += $(CPPFLAGS)

# Kernel source
SRC    := kernel.cpp
TARGET := tk_mha_onekernel_bwd

# Log directory
LOGDIR  := build_logs
LOGFILE := $(LOGDIR)/build.log

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(PY_EXT_SUFFIX) 2>&1 | tee $(LOGFILE)

clean:
	rm -f $(TARGET)$(PY_EXT_SUFFIX) $(TARGET)*.so
	rm -rf $(LOGDIR)

# Convenience: compile-only check (no linking)
check: $(SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) -fsyntax-only $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) 2>&1 | tee $(LOGFILE)

.PHONY: all clean check
