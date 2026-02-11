# Unified Attention Forward Kernel Makefile
GPU_TARGET=CDNA4

TARGET=tk_unified_attn

# Parameters
NUM_Q_HEADS ?= 32
NUM_KV_HEADS ?= 8
HEAD_SIZE ?= 128
BLOCK_SIZE ?= 16
TILE_SIZE ?= 64
SLIDING_WINDOW ?= 0
USE_ALIBI_SLOPES ?= 0
USE_SOFTCAP ?= 0

SRC=kernel.cpp

CPPFLAGS += -DNUM_Q_HEADS=$(NUM_Q_HEADS)
CPPFLAGS += -DNUM_KV_HEADS=$(NUM_KV_HEADS)
CPPFLAGS += -DHEAD_SIZE=$(HEAD_SIZE)
CPPFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)
CPPFLAGS += -DTILE_SIZE=$(TILE_SIZE)
CPPFLAGS += -DSLIDING_WINDOW=$(SLIDING_WINDOW)
CPPFLAGS += -DUSE_ALIBI_SLOPES=$(USE_ALIBI_SLOPES)
CPPFLAGS += -DUSE_SOFTCAP=$(USE_SOFTCAP)

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS+= -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

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

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) -shared -fPIC -Rpass-analysis=kernel-resource-usage --save-temps

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)*.so *.o *.bc *.s *.ii
