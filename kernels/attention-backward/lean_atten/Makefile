# Lean Attention Forward Kernel - Makefile
# Follows the reference HipKittens build pattern

# GPU Target: CDNA3 (MI300X) or CDNA4 (MI350X)
GPU_TARGET ?= CDNA4

TARGET = lean_attn_kernel
SRC    = kernel.cpp

# Configurable parameters (passed as -D flags)
ATTN_B   ?= 4
ATTN_H   ?= 32
ATTN_N   ?= 1024
ATTN_D   ?= 128

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_N=$(ATTN_N)
CPPFLAGS += -DATTN_D=$(ATTN_D)

# HIP / ROCm paths
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# GPU-specific compiler flags
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# C++20 standard
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
ICXXFLAGS  += $(PY_LDFLAGS)
ICXXFLAGS  += -I${THUNDERKITTENS_ROOT}/include \
              -I${THUNDERKITTENS_ROOT}/prototype \
              $(shell python3 -m pybind11 --includes) \
              -shared -fPIC \
              -Rpass-analysis=kernel-resource-usage

# Default target
all: $(TARGET)

LOGDIR  := /workdir/data_logs/$(shell date +%m%d_%H%M%S)_outputs
LOGFILE := $(LOGDIR)/make_build.log

$(TARGET): $(SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix) 2>&1 | tee $(LOGFILE)

clean:
	rm -f $(TARGET)*.so

.PHONY: all clean
