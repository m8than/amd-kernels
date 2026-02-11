HIPCC ?= hipcc
HIPCC_FLAGS = -std=c++20 -O3 -fPIC
HIPCC_FLAGS += -munsafe-fp-atomics
HIPCC_FLAGS += --offload-arch=gfx942

TARGET = fused_recurrent.so
SRC = kernel.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -shared -o $@ $<

clean:
	rm -f $(TARGET)
