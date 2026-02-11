# Softmax HipKittens Kernel Makefile

GPU_TARGET=CDNA3

TARGET=softmax_kernel
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
ICPPFLAGS := -I$(HIP_INCLUDE_DIR)
ICXXFLAGS += -shared -fPIC

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET).so

clean:
	rm -f $(TARGET)*.so
