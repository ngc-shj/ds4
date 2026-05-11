CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)

# DS4_BACKEND selects the GPU backend on Linux: cpu (default reference path),
# cuda (NVIDIA CUDA, intended for GB10/Grace Blackwell DGX Spark and similar).
# Darwin always builds the Metal backend regardless of this variable.
DS4_BACKEND ?= cpu

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
BACKEND_LDLIBS := $(METAL_LDLIBS)
else
# Linux/POSIX: c99 hides clock_gettime, sigaction, PATH_MAX, fileno, etc.
# _GNU_SOURCE re-enables them without forcing every TU to declare macros.
CFLAGS += -D_GNU_SOURCE
ifeq ($(DS4_BACKEND),cuda)
NVCC ?= nvcc
# GB10 (DGX Spark) is sm_121; keep sm_90 too so the binary still runs on
# Hopper-class developer machines. -O3 is the nvcc default at -O3 host flag.
NVCCFLAGS ?= -O3 --use_fast_math -std=c++17 \
             -gencode arch=compute_90,code=sm_90 \
             -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_121,code=sm_121
CFLAGS += -DDS4_CUDA
CORE_OBJS = ds4.o ds4_cuda.o
NATIVE_CORE_OBJS = ds4_native.o
# CUDA runtime only.  Driver API not needed; cuBLAS will be linked when the
# matmul kernels are wired up.
CUDA_PATH ?= /usr/local/cuda
# -lstdc++ pulls the C++ runtime that nvcc-emitted host code (and the CUDA
# fatbin registration glue) depends on; ds4 itself stays plain C.
# cuBLAS provides the F16/F32 GEMMs used by ds4_metal_matmul_f{16,32}_tensor.
BACKEND_LDLIBS := $(LDLIBS) -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lstdc++
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
BACKEND_LDLIBS := $(LDLIBS)
endif
METAL_LDLIBS := $(BACKEND_LDLIBS)
endif

.PHONY: all clean test

all: ds4 ds4-server

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(BACKEND_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(BACKEND_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

# CUDA backend (Linux + DS4_BACKEND=cuda).  ds4_cuda.cu mirrors ds4_metal.m's
# public surface (ds4_metal_* symbols) and #includes the kernel sources from
# cuda/*.cu, so ds4.c can keep its existing call sites.
ds4_cuda.o: ds4_cuda.cu ds4_metal.h $(wildcard cuda/*.cu) $(wildcard cuda/*.cuh)
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4_native ds4_server_test ds4_test *.o
