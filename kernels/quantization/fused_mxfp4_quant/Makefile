# Makefile for fused RMSNorm + MXFP4 quantization kernel

HIPKITTENS_ROOT ?= ../../HipKittens
HIPCC ?= hipcc
CXXFLAGS = -std=c++20 -O3 -fPIC -I$(HIPKITTENS_ROOT)/include
LDFLAGS = -shared -fPIC

TARGET = libfused_mxfp4_quant.so

all: $(TARGET)

$(TARGET): kernel.cpp
	$(HIPCC) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python test_parity.py

.PHONY: all clean test
