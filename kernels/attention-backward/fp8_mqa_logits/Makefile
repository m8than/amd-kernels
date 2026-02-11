# Makefile for FP8 MQA Logits HipKittens kernel
#
# Usage:
#   make                         # build with defaults
#   make NUM_HEADS=16 HEAD_SIZE=128 SEQ_LEN=512 SEQ_LEN_KV=1024
#   make clean

# Compiler target -- default to CDNA4 (MI350X)
GPU_TARGET ?= CDNA4

TARGET = tk_fp8_mqa_logits
SRC    = kernel.cpp

# Kernel configuration (override on command line or via environment)
NUM_HEADS  ?= 8
HEAD_SIZE  ?= 64
SEQ_LEN    ?= 256
SEQ_LEN_KV ?= 256

CPPFLAGS += -DNUM_HEADS=$(NUM_HEADS)
CPPFLAGS += -DHEAD_SIZE=$(HEAD_SIZE)
CPPFLAGS += -DSEQ_LEN=$(SEQ_LEN)
CPPFLAGS += -DSEQ_LEN_KV=$(SEQ_LEN_KV)

# ROCm / HIP paths
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# GPU-target-specific flags
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# C++ standard and include paths
CXX_STD    := c++20
ICXXFLAGS  := -std=$(CXX_STD) -w
ICPPFLAGS  := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS   :=
ILDLIBS    :=

ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

# Python / pybind11
PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS  += $(PY_LDFLAGS)
ICXXFLAGS  += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
              $(shell python3 -m pybind11 --includes) -shared -fPIC \
              -Rpass-analysis=kernel-resource-usage --save-temps

# Build targets
all: $(TARGET)

LOGDIR  := /workdir/data_logs/$(shell date +%m%d_%H%M%S)_outputs
LOGFILE := $(LOGDIR)/make_build.log

$(TARGET): $(SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix) 2>&1 | tee $(LOGFILE)

clean:
	rm -f $(TARGET)*.so $(TARGET)*.ii $(TARGET)*.s $(TARGET)*.bc
