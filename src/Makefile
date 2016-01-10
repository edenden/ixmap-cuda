OBJS = main.o \
fib.o fib_cuda.o \
lpm.o lpm_cuda.o \
neigh.o neigh_cuda.o \
hash.o hash_cuda.o \
forward.o forward_cuda.o \
epoll.o netlink.o thread.o iftap.o misc.o

TARGET = a.out
CUDA_PATH = /usr/local/cuda

NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE_DIR = $(CUDA_PATH)/include/
CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib64/

CFLAGS = -I$(CUDA_INCLUDE_DIR) -I../lib/include/
CCFLAGS = -O2 -Wall
LDFLAGS = -L$(CUDA_LIBRARY_DIR) -lcuda -lcudart -lpthread -lixmap -lnuma

all: $(TARGET)

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) ../lib/ixmap_cuda.a $(LDFLAGS) -o $(TARGET)

.SUFFIXES: .c .cu .o

.cu.o:
	$(NVCC) $(CFLAGS) --x cu --compile --relocatable-device-code=true $< -o $@

.c.o:
	$(CC) $(CFLAGS) $(CCFLAGS) -c $< -o $@
