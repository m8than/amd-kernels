# Makefile for MoE Routing Sigmoid Top-1 Fused Kernel

# Compiler and flags
HIPCC = hipcc
CXXFLAGS = -std=c++20 -O3 -fPIC
HIPFLAGS = --offload-arch=gfx942 --offload-arch=gfx950
INCLUDES = -I../../reference/hipkittens/include

# Output
TARGET = libmoe_routing_sigmoid_top1.so
KERNEL_SRC = kernel.cpp

# Build shared library
all: $(TARGET)

$(TARGET): $(KERNEL_SRC)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -shared -o $@ $<

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all clean
