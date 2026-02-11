# Makefile for MHA Fused Backward Pass - HipKittens kernel
#
# Usage:
#   make                     # build the kernel module
#   make GPU_TARGET=CDNA3    # target gfx942 (MI300X)
#   make GPU_TARGET=CDNA4    # target gfx950 (MI350X)
#   make clean               # remove build artifacts
#
# Parameters (override at build time):
#   ATTN_B=4 ATTN_H=32 ATTN_H_KV=8 ATTN_N=512

# GPU target architecture
GPU_TARGET ?= CDNA3

# Attention parameters
ATTN_B    ?= 4
ATTN_H    ?= 32
ATTN_H_KV ?= 8
ATTN_N    ?= 512

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

# ROCm / HIP paths
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX           ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# GPU-specific flags
ifeq ($(GPU_TARGET),CDNA2)
  HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
  HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
  HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# C++ standard and common flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD) -w
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)

# Python / pybind11
PY_LDFLAGS := $(shell python3-config --ldflags 2>/dev/null | sed 's/-lcrypt//g')
PY_INCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null)
PY_EXT_SUFFIX := $(shell python3-config --extension-suffix 2>/dev/null)

ICXXFLAGS += $(PY_LDFLAGS) $(PY_INCLUDES) -shared -fPIC
ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include
ICXXFLAGS += -Rpass-analysis=kernel-resource-usage

# Source and target
SRC    := kernel.cpp
TARGET := mha_fused_bwd

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(CPPFLAGS) \
		-o $(TARGET)$(PY_EXT_SUFFIX)

clean:
	rm -f $(TARGET)*.so $(TARGET)*.cpython*

.PHONY: all clean
