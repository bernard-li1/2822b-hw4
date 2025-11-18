# Compiler to use
CXX = clang++
MPICXX = mpicxx
NVCC = nvcc

# Compilation flags
# -O3 for optimization
CXXFLAGS = -Wall -Wextra -std=c++17 -O3
# Flags for OpenMP (Retained for reference)
OMPFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp

# CUDA Compilation flags
# May need to change -arch to match specific GPU architecture. Using H100 on Oscar.
NVCCFLAGS = -O3 -std=c++17 -arch=sm_90

# Source files
TASK1_SRC = task_1.cu
TASK2_SRC = task_2.cu

# Executables
EXECS = task_1 task_2

all: $(EXECS)

# Rule to build task_1 (Pure CUDA application)
task_1: $(TASK1_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Rule to build task_2 (MPI-CUDA Hybrid application)
# We use MPICXX to compile/link, passing CUDA flags to NVCC via -Xcompiler
# This ensures that both MPI and CUDA libraries are linked correctly.
task_2: $(TASK2_SRC)
	$(MPICXX) $(NVCCFLAGS) -x cu -o $@ $< -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f $(EXECS) *.o

.PHONY: all clean