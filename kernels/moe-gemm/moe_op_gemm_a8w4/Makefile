HIPCC ?= hipcc
HIPKITTENS_ROOT ?= ../../reference/hipkittens
INCLUDE_DIR = $(HIPKITTENS_ROOT)/include
CXXFLAGS = -std=c++20 -O3 --offload-arch=gfx942 -I$(INCLUDE_DIR)

TARGET = moe_op_gemm_a8w4_kernel

.PHONY: all clean

all: $(TARGET)

$(TARGET): kernel.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
