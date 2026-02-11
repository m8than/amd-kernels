HIPCC ?= hipcc
HIPCC_FLAGS = -std=c++20 -O3 -fPIC
HIPCC_FLAGS += -munsafe-fp-atomics
HIPCC_FLAGS += --offload-arch=gfx942

TARGET = causal_conv1d_split_qkv.so
SRC = kernel.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -shared -o $@ $<

clean:
	rm -f $(TARGET)
