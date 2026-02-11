HIPCC ?= hipcc
HIPKITTENS_DIR ?= ../../reference/hipkittens/include
CXXFLAGS = -std=c++20 -O3 -I$(HIPKITTENS_DIR) --offload-arch=gfx942

TARGET = kernel

all: $(TARGET)

$(TARGET): kernel.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
