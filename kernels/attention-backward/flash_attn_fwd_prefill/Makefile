# Flash Attention Forward Prefill - HipKittens Build
#
# Usage:
#   make                       # Build with default parameters
#   make ATTN_B=8 ATTN_N=2048  # Override parameters
#   make clean                 # Remove build artifacts
#
# Requires:
#   - ROCm/HIP installation (hipcc)
#   - THUNDERKITTENS_ROOT environment variable pointing to HipKittens include root
#   - Python 3 with pybind11

# ============================================================================
# GPU Target
# ============================================================================
GPU_TARGET ?= CDNA4

# ============================================================================
# Module name
# ============================================================================
TARGET = flash_attn_fwd_prefill
SRC = kernel.cpp

# ============================================================================
# Kernel parameters (overridable)
# ============================================================================
ATTN_B   ?= 4
ATTN_H   ?= 32
ATTN_H_KV ?= 8
ATTN_N   ?= 1024
ATTN_D   ?= 128
IS_CAUSAL ?= 1

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)
CPPFLAGS += -DATTN_D=$(ATTN_D)

ifeq ($(IS_CAUSAL),1)
CPPFLAGS += -DIS_CAUSAL=true
else
CPPFLAGS += -DIS_CAUSAL=false
endif

# ============================================================================
# ROCm / HIP paths
# ============================================================================
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX           ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# ============================================================================
# GPU architecture flags
# ============================================================================
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# ============================================================================
# Compiler flags
# ============================================================================
CXX_STD    := c++20
ICXXFLAGS  := -std=$(CXX_STD)
ICPPFLAGS  := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS   :=
ILDLIBS    :=

# Suppress warnings for cleaner build output
CXXFLAGS   := -w

ICXXFLAGS  += $(CXXFLAGS)
ICPPFLAGS  += $(CPPFLAGS)
ILDFLAGS   += $(LDFLAGS)
ILDLIBS    += $(LDLIBS)

# Python / pybind11 flags
PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS  += $(PY_LDFLAGS)
ICXXFLAGS  += -I${THUNDERKITTENS_ROOT}/include \
              -I${THUNDERKITTENS_ROOT}/prototype \
              $(shell python3 -m pybind11 --includes) \
              -shared -fPIC \
              -Rpass-analysis=kernel-resource-usage

# ============================================================================
# Build targets
# ============================================================================
EXTENSION_SUFFIX := $(shell python3-config --extension-suffix)

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $(TARGET)$(EXTENSION_SUFFIX)

clean:
	rm -f $(TARGET)$(EXTENSION_SUFFIX)
	rm -f *.o

.PHONY: all clean
