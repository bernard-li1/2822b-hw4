# Compiler setup
# Use hipcc for compiling HIP/device code
HIPCC = hipcc
MPICXX = mpicxx

# Compilation flags
# --offload-arch=native tells hipcc to target the GPU architecture of the current machine (e.g., MI250X, MI300)
# -std=c++17 is required for newer C++ features
CXXFLAGS = -O3 -std=c++17 --offload-arch=native -Wall -Wextra

# Source files
# Updated to match the .cc extensions in your uploaded files
TASK1_SRC = task_1.cc
TASK2_SRC = task_2.cc

# Executables
EXECS = task_1 task_2

all: $(EXECS)

# Rule to build task_1 (Single GPU, Pure HIP)
# We use hipcc directly.
task_1: $(TASK1_SRC)
	$(HIPCC) $(CXXFLAGS) -o $@ $<

# Rule to build task_2 (MPI + HIP)
# Since the code contains kernels (<<<>>>), we must compile with hipcc.
# We use mpicxx --showme (or -show) to get the necessary MPI includes and libraries 
# and pass them to hipcc.
MPI_COMPILE_FLAGS = $(shell $(MPICXX) --showme:compile 2>/dev/null || $(MPICXX) -show -c | sed 's/^[^ ]* //')
MPI_LINK_FLAGS    = $(shell $(MPICXX) --showme:link 2>/dev/null || $(MPICXX) -show -l | sed 's/^[^ ]* //')

task_2: $(TASK2_SRC)
	$(HIPCC) $(CXXFLAGS) $(MPI_COMPILE_FLAGS) -o $@ $< $(MPI_LINK_FLAGS)

clean:
	rm -f $(EXECS) *.o

.PHONY: all clean