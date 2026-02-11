# Fused QKV Split + QK RoPE HipKittens Kernel Makefile

GPU_TARGET=CDNA3

TARGET=fused_qkv_rope_kernel
SRC=kernel.cpp

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

CXX_STD := c++20
ICXXFLAGS := -std=$(CXX_STD) -O3 -w
ICPPFLAGS := -I../../reference/hipkittens/include -I$(HIP_INCLUDE_DIR)

PY_LDFLAGS := $(shell python3-config --ldflags 2>/dev/null | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS += -I../../reference/hipkittens/include $(shell python3 -m pybind11 --includes 2>/dev/null) -shared -fPIC

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix 2>/dev/null || echo .so)

clean:
	rm -f $(TARGET)*.so
