OBJS = main.o fib.o lpm.o fib_cuda.o lpm_cuda.o
BIN = a.out
CUDA_PATH = /usr/local/cuda

NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_INCLUDE_DIR = $(CUDA_PATH)/include/
CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib64/

CFLAGS = -I$(CUDA_INCLUDE_DIR)
CCFLAGS = -O2 -Wall
LDFLAGS = -L$(CUDA_LIBRARY_DIR) -lcuda -lcudart 

all: $(BIN)

clean:
	rm $(OBJS)
	rm $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(CCFLAGS) $(LDFLAGS) $(OBJS) -o $(BIN)

.SUFFIXES: .c .cu .o

.cu.o:
	$(NVCC) $(CFLAGS) --x cu --compile $< -o $@

.c.o:
	$(CC) $(CFLAGS) $(CCFLAGS) -c $< -o $@
