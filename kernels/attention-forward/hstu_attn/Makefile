# HSTU Attention Forward Kernel Makefile
GPU_TARGET=CDNA4

TARGET=tk_hstu_attn

# Parameters
ATTN_H ?= 16
BLOCK_D_Q ?= 128
BLOCK_D_V ?= 128
IS_CAUSAL ?= 1
HAS_MULTIPLE_TARGETS ?= 0
HAS_CONTEXTUAL_SEQ_LEN ?= 0
HAS_MAX_ATTN_LEN ?= 0

SRC=kernel.cpp

CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DBLOCK_D_Q=$(BLOCK_D_Q)
CPPFLAGS += -DBLOCK_D_V=$(BLOCK_D_V)
CPPFLAGS += -DIS_CAUSAL=$(IS_CAUSAL)
CPPFLAGS += -DHAS_MULTIPLE_TARGETS=$(HAS_MULTIPLE_TARGETS)
CPPFLAGS += -DHAS_CONTEXTUAL_SEQ_LEN=$(HAS_CONTEXTUAL_SEQ_LEN)
CPPFLAGS += -DHAS_MAX_ATTN_LEN=$(HAS_MAX_ATTN_LEN)

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
