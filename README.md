## README: 2D Jacobi Solver Implementations

This document describes the algorithms implemented in `task_1.cu` (Pure CUDA) and `task_2.cu` (MPI + CUDA) for solving the 2D Poisson equation using the **Jacobi iterative method**. The problem is defined on a 2D domain with a Dirichlet boundary condition based on the exact solution $u(x, y) = \sin(2\pi x) \sin(2\pi y)$.

---

## 1. `task_1.cu`: Pure CUDA Jacobi Solver

This file implements the 2D Jacobi iterative method entirely on a single CUDA-enabled GPU.

### 1.1 Algorithm Overview

The Jacobi method is used to solve the discretized Poisson equation $\nabla^2 u = f$ iteratively. The update rule for an interior point $u_{i,j}$ is:

$$u_{i, j}^{k+1} = \frac{1}{4} (u_{i+1, j}^k + u_{i-1, j}^k + u_{i, j+1}^k + u_{i, j-1}^k - h^2 f_{i, j})$$

The solver iterates until the maximum absolute difference between $u^{k+1}$ and $u^k$ across the entire grid (the residual) falls below a specified **tolerance** ($\text{TOLERANCE}$).

### 1.2 Key CUDA Implementations

#### 1. **2D Tiling and Shared Memory (SMEM) in `jacobi` Kernel**

The `jacobi` kernel uses a 2D block grid and thread block configuration (defined by `TILE_X` and `TILE_Y`) to enable efficient data reuse via **Shared Memory (SMEM)**.

* **Tiling:** Each thread block is responsible for updating a tile of the grid.
* **Halo/Ghost Cells:** Since the Jacobi stencil (5-point cross) requires values from neighboring cells, the SMEM tile is allocated with a $1$-cell **halo** around the core data. For a tile of size $T_Y \times T_X$, the shared memory array is $(T_Y + 2) \times (T_X + 2)$.
* **Data Loading:** All threads in the block cooperate to load the necessary $u$ values (including the halo) from global memory (HBM) into SMEM.
* **Computation:** Once the data is loaded and synchronized (`__syncthreads()`), each thread computes its new value $u_{new}$ using the fast SMEM access, significantly reducing global memory traffic.

#### 2. **Residual Calculation and Reduction**

To check for convergence, the maximum absolute difference ($\text{MaxDiff}$) between the old and new grid must be calculated after each iteration.

* **Local Block Max:** Inside the `jacobi` kernel, the `cub::BlockReduce` library function is used to efficiently find the maximum difference (`threaddiff`) within each thread block. This result is written to a global array, `d_partial_max`.
* **Global Max Reduction:** The `reduce_final_max` host helper function performs a parallel, multi-stage reduction on the partial maxima. It launches the `block_reduce` kernel iteratively until a single global maximum difference is obtained. This reduction uses a **ping-pong buffering** approach with `d_partial_max` and `d_tmp1`.

#### 3. **Memory Management**

* **Device Memory:** `d_u`, `d_unew`, `d_f`, and `d_u_exact` are allocated on the GPU. The solution is swapped between `d_u` and `d_unew` each iteration, eliminating a memory copy.
* **Initialization:** The `init_grids` kernel initializes the forcing function $f$, the exact solution $u_{exact}$, and applies Dirichlet boundary conditions (Dirichlet BCs) based on $u_{exact}$.

---

## 2. `task_2.cu`: MPI + CUDA Jacobi Solver (Hybrid Parallelism)

This file implements a hybrid parallel solution using **MPI** for inter-node/inter-GPU communication and **CUDA** for intra-node/intra-GPU parallelism. The domain is decomposed in the **Y-direction (rows)** across the MPI ranks.

### 2.1 Algorithm Overview and Domain Decomposition

* **Parallelization Strategy:** The $N \times N$ grid is logically divided into $R$ horizontal strips, where $R$ is the number of MPI ranks (`size`). Each MPI rank handles its own local grid strip.
* **Ghost Cells:** To compute the stencil at the boundary of a local strip, a rank requires data from its neighbor's adjacent row. The local grid is allocated with two extra rows—a **top ghost row (local row 0)** and a **bottom ghost row (local row local\_N+1)**—to store data received from the neighboring ranks.
* **Communication:** Before each Jacobi iteration, ranks must exchange their topmost and bottommost *owned* rows with their neighbors.

### 2.2 Key Hybrid Implementations

#### 1. **MPI Communication (`MPI_Isend` / `MPI_Irecv`)**

The data exchange is handled using **non-blocking MPI calls** to potentially overlap computation and communication.

* **Send:** Owned rows (local rows 1 and $\text{local\_N}$) are copied from device memory (`d_u`) to **pinned host buffers** (`h_send_buffer_top`, etc.) using `cudaMemcpyDeviceToHost`, and then sent using `MPI_Isend`.
* **Receive:** Incoming ghost rows are received into pinned host buffers (`h_recv_buffer_top`, etc.) using `MPI_Irecv`.
* **Synchronization:** `MPI_Waitall` is called to ensure all data has been transferred.
* **Update Device Ghost Rows:** The received ghost row data is copied from the host buffers to the device ghost rows (local rows 0 and $\text{local\_N}+1$) using `cudaMemcpyHostToDevice`. This copy is **skipped** if the rank is at a global boundary (i.e., `rank_up` or `rank_down` is `MPI_PROC_NULL`), preserving the Dirichlet BCs.

#### 2. **Local CUDA Kernels**

The core kernels are adapted to operate only on the local, padded grid strip.

* **`init_local_grids`:** Initializes the owned rows ($\text{local\_row} \in [1, \text{local\_N}]$) and applies Dirichlet BCs to any global boundaries falling within the strip.
* **`jacobi_mpi`:** The 2D tiling/SMEM optimization is used. The kernel only updates **owned interior points**. The stencil operation reads from the SMEM tile, which includes the up-to-date ghost rows transferred via MPI.

#### 3. **Global Convergence and Error Check**

* **Local Max Reduction:** The `jacobi_mpi` kernel computes the local max difference for the rank's strip, reduced on the device using `reduce_final_max`.
* **Global Reduction:** The local maximum difference/error from all ranks is aggregated using `MPI_Allreduce` (for convergence check) or `MPI_Reduce` (for final error check) with the $\text{MPI\_MAX}$ operation to find the true global maximum residual/error.

### 2.3 Other Features (Common to both tasks)

* **`restrict` keyword:** Used in kernel signatures (`double* __restrict__`) to improve compiler optimization by asserting that pointers do not alias.
* **CUDA Error Handling:** The `checkCudaErrors` macro ensures that all CUDA API calls and kernel launches are checked for errors.
* **CUB Library:** `cub::BlockReduce` is used in the Jacobi kernels for highly efficient, warp-optimized local reduction of the MaxDiff value.
