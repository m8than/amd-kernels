GPU_TARGET ?= CDNA4
TARGET = tk_fused_gemm_afp4wfp4_split_cat
SRC = kernel.cpp

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := /opt/rocm/include/hip
HIPCXX ?= /opt/rocm/bin/hipcc

ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950
endif

CXX_STD := c++20
ICXXFLAGS := -std=$(CXX_STD) -w
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
    $(shell python3 -m pybind11 --includes) -shared -fPIC \
    -Rpass-analysis=kernel-resource-usage

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)*.so
