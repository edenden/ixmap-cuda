OBJS = ixmap.o ixmap_cuda.o
TARGET = ixmap_cuda.a
CUDA_PATH = /usr/local/cuda

AR = ar
ARFLAGS = rv
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE_DIR = $(CUDA_PATH)/include/
CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib64/

CFLAGS = -I$(CUDA_INCLUDE_DIR)
CCFLAGS = -O2 -Wall

all: $(TARGET)

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

.SUFFIXES: .c .cu .o

.cu.o:
	$(NVCC) $(CFLAGS) --x cu --compile --relocatable-device-code=true $< -o $@

.c.o:
	$(CC) $(CFLAGS) $(CCFLAGS) -c $< -o $@
