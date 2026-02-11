# LayerNorm HipKittens kernel build
GPU_TARGET=CDNA3

TARGET=layernorm_tk
SRC=kernel.cpp

# HipKittens include path (relative to this directory)
HK_INCLUDE=../../reference/hipkittens/include

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I/opt/rocm/include/rocrand
endif

# Common flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD) -w -O3
ICPPFLAGS := -I$(HK_INCLUDE) -I$(HIP_INCLUDE_DIR)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS += -I$(HK_INCLUDE) $(shell python3 -m pybind11 --includes) -shared -fPIC

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)*.so
