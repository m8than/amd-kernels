# Flash Attention v2 Backward Pass - HipKittens Makefile
#
# Usage:
#   make                        # Build for default GPU target (CDNA4)
#   make GPU_TARGET=CDNA3       # Build for MI300X (gfx942)
#   make GPU_TARGET=CDNA2       # Build for MI250 (gfx90a)
#   make clean                  # Remove build artifacts
#
# Parameters (overridable):
#   ATTN_B    - batch size (default: 4)
#   ATTN_H    - number of query heads (default: 32)
#   ATTN_H_KV - number of KV heads (default: 8)
#   ATTN_N    - sequence length (default: 1024)

# GPU target
GPU_TARGET ?= CDNA4

# Attention parameters
ATTN_B    ?= 4
ATTN_H    ?= 32
ATTN_H_KV ?= 8
ATTN_N    ?= 1024

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

# ROCm / HIP
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

# C++ standard and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

# Suppress warnings (HipKittens headers generate many)
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

# Output extension
PY_EXT := $(shell python3-config --extension-suffix)

# Build target name
TARGET := flash_attn_bwd

# ============================================================================
# Targets
# ============================================================================

all: $(TARGET)

$(TARGET): kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $@$(PY_EXT)

clean:
	rm -f $(TARGET)*.so $(TARGET)$(PY_EXT)

.PHONY: all clean
