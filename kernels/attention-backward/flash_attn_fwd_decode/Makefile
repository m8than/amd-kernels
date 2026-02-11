# Flash Attention Forward Decode - HipKittens Split-K
# Compiler: hipcc with C++20

GPU_TARGET ?= CDNA4
TARGET = flash_attn_fwd_decode
SRC = kernel.cpp

# HIP / ROCm configuration
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

# C++ standard and warnings
CXX_STD := c++20
ICXXFLAGS := -std=$(CXX_STD) -w

# Include paths: HipKittens headers
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ICPPFLAGS += -I${THUNDERKITTENS_ROOT}/prototype

# pybind11 integration
PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PY_EXT_SUFFIX := $(shell python3-config --extension-suffix)

ICXXFLAGS += $(PY_LDFLAGS) $(PYBIND11_INCLUDES)
ICXXFLAGS += -shared -fPIC

# Build targets
all: $(TARGET)

LOGDIR := /workdir/data_logs/$(shell date +%m%d_%H%M%S)_outputs
LOGFILE := $(LOGDIR)/make_build.log

$(TARGET): $(SRC)
	@mkdir -p $(LOGDIR) 2>/dev/null || true
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(PY_EXT_SUFFIX) 2>&1 | tee $(LOGFILE) || true

clean:
	rm -f $(TARGET)$(PY_EXT_SUFFIX)
	rm -f *.ii *.i *.bc *.s *.o

.PHONY: all clean
