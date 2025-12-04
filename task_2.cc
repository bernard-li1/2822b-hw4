#include <iostream>
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iomanip>
#include <stdio.h>
#include <algorithm>

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

using namespace std;

// --- HIP Error Checking ---
#define checkHipErrors(val) check_hip( (val), #val, __FILE__, __LINE__ )
inline void check_hip(hipError_t result, const char *const func, const char *const file, const int line) {
    if (result != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d, %s at '%s'\n", file, line, hipGetErrorString(result), func);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
}

// --- Kernel Configuration ---
#ifndef TILE_X
#define TILE_X 16
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif 

/**
 * @brief Initialize local grids. 
 * Sets Dirichlet BCs on global boundaries and ghost rows.
 */
__global__ void init_local_grids(double* __restrict__ u, double* __restrict__ unew, double* __restrict__ f, double* __restrict__ u_exact, 
                                 int N, double h, int local_N, int global_start_row) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i_local = blockIdx.y * blockDim.y + threadIdx.y;

    // Only initialize rows we own (1 to local_N)
    if (i_local >= 1 && i_local <= local_N && j < N) {
        
        int i_global = global_start_row + i_local - 1;
        double x = i_global * h;
        double y = j * h;
        int idx_local = i_local * N + j;

        u_exact[idx_local] = sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
        f[idx_local] = -8.0 * M_PI * M_PI * sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);

        // Apply Dirichlet (fixed) BCs for Interior points touching global borders
        if (i_global == 0 || i_global == N-1 || j == 0 || j == N-1) {
            u[idx_local] = u_exact[idx_local];
            unew[idx_local] = u_exact[idx_local];
        } else {
            u[idx_local] = 0.0;
            unew[idx_local] = 0.0;
        }
    }
}

__global__ void jacobi_mpi(const double* __restrict__ u, double* __restrict__ unew,
                           const double* __restrict__ f, double* __restrict__ d_partial_max, 
                           int N, double h, int local_N, int global_start_row) {
    
    int j = blockIdx.x * TILE_X + threadIdx.x;
    int i = blockIdx.y * TILE_Y + threadIdx.y; 
    
    extern __shared__ double smem[];
    int scols = TILE_X + 2;
    double* s_tile = smem; 

    int total_cells = (TILE_Y + 2) * scols; 
    const int nthreads = TILE_X * TILE_Y;

    // Collaborative loading into shared memory
    for (int k=0; k<total_cells; k += nthreads) {
        int s_i = (threadIdx.y * TILE_X + threadIdx.x) + k;
        if (s_i < total_cells) {
            int local_i_load = blockIdx.y * TILE_Y  + (s_i / scols)-1;
            int local_j_load = blockIdx.x * TILE_X + (s_i % scols)-1;

            double val = 0.0; 
            if (local_i_load >= 0 && local_i_load < (local_N + 2) && local_j_load >= 0 && local_j_load < N) {
                val = u[local_i_load * N + local_j_load]; 
            }
            s_tile[s_i] = val;
        }
    }
    __syncthreads(); 

    double threaddiff = 0.0;
    
    // Update only owned INTERIOR points
    if (i >= 1 && i <= local_N && j > 0 && j < N-1) {
        int i_global = global_start_row + i - 1;

        // STRICT DIRICHLET: Do not update global boundaries (0 and N-1)
        if (i_global > 0 && i_global < N-1) {
            
            int s_i = threadIdx.y + 1; 
            int s_j = threadIdx.x + 1;
            int idx_local = i * N + j; 

            double up = s_tile[(s_i - 1) * scols + s_j];
            double down = s_tile[(s_i+1)*scols+s_j];
            double left = s_tile[s_i * scols + (s_j-1)];
            double right = s_tile[s_i * scols + (s_j+1)];
            
            double val = 0.25 * (up + down + left + right - h * h * f[idx_local]);
            unew[idx_local] = val;
            threaddiff = fabs(val - s_tile[s_i * scols + s_j]);
        }
    }

    // --- Block Reduce ---
    using BlockReduce = hipcub::BlockReduce<double, TILE_X * TILE_Y>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Define a custom functor to avoid "Max" macro collisions from mpi.h
    struct MaxOp {
        __device__ __forceinline__
        double operator()(const double &a, const double &b) const {
            return (a > b) ? a : b;
        }
    };

    double block_max = BlockReduce(temp_storage).Reduce(threaddiff, MaxOp());

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        d_partial_max[blockIdx.y * gridDim.x + blockIdx.x] = block_max;
    }
}

__global__ void check_soln_mpi(const double* __restrict__ u, const double* __restrict__ u_exact,
                               double* __restrict__ d_partial_max_error, const int N, const int local_N) {
    
    extern __shared__ double s_data[];
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int block_size = blockDim.x * blockDim.y;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    double err = 0.0;
    if (i >= 1 && i <= local_N && j < N) {
        int idx = i * N + j;
        err = fabs(u_exact[idx] - u[idx]);
    }
    s_data[tid] = err;
    __syncthreads();

    // Standard tree reduction in shared memory
    for (unsigned int s = block_size / 2; s>0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmax(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        d_partial_max_error[block_idx] = s_data[0];
    }
}

#define FINAL_RED_BLOCK_SZ 256
__global__ void block_reduce(const double* __restrict__ d_partial, double* __restrict__ d_out, int M) {
    __shared__ double sdata[FINAL_RED_BLOCK_SZ];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    double tmax = 0.0;
    
    if (i < (unsigned int)M) tmax = d_partial[i];
    if (i + blockDim.x < (unsigned int)M) {
        double v = d_partial[i + blockDim.x];
        if (v > tmax) tmax = v;
    }
    sdata[tid] = tmax;
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

double reduce_final_max(double* d_partial, int num_partials, double* d_tmp1, double* h_result_ptr) {
    int M = num_partials;
    int threads = FINAL_RED_BLOCK_SZ;
    double* d_in = d_partial;
    double* d_out = d_tmp1;

    do {
        int blocks = (M + threads*2 - 1) / (threads*2);
        block_reduce<<<blocks, threads>>>(d_in, d_out, M);
        checkHipErrors(hipGetLastError());
        std::swap(d_in, d_out);
        M = blocks;
    } while (M > 1);

    checkHipErrors(hipMemcpy(h_result_ptr, d_in, sizeof(double), hipMemcpyDeviceToHost));
    return *h_result_ptr;
}

int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 201;
    if (argc >= 2) N = atoi(argv[1]);
    int MAX_ITER = 100000;
    if (argc >= 3) MAX_ITER = atoi(argv[2]);

    const double h = 1.0 / (N-1);
    const double TOLERANCE = 1.0e-8;
    const int PRINT_FREQ = 500;

    if (rank == 0) {
        cout << "Running MPI+HIP Jacobi solver for N = " << N << " with max iterations = " << MAX_ITER << " on " << size << " ranks." << endl;
    }

    int nDevices = 0;
    checkHipErrors(hipGetDeviceCount(&nDevices));
    if (nDevices == 0) {
        if (rank == 0) fprintf(stderr, "No HIP devices found!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device = rank % nDevices; 
    checkHipErrors(hipSetDevice(device));
    if (rank == 0) cout << "Total HIP devices found: " << nDevices << endl;
    printf("Rank %d is using GPU %d\n", rank, device);

    // --- Domain Decomposition ---
    int rows_per_proc = N / size;
    int remainder_rows = N % size;
    int local_N = (rank < remainder_rows) ? (rows_per_proc + 1) : rows_per_proc;
    
    int global_start_row = 0;
    if (rank < remainder_rows) {
        global_start_row = rank * (rows_per_proc + 1);
    } else {
        global_start_row = remainder_rows * (rows_per_proc + 1) + (rank - remainder_rows) * rows_per_proc;
    }
    
    int rank_up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int rank_down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    size_t local_padded_rows = local_N + 2;
    size_t local_grid_size = local_padded_rows * N * sizeof(double);
    
    double *d_u = nullptr, *d_unew = nullptr, *d_f = nullptr, *d_u_exact = nullptr;
    checkHipErrors(hipMalloc(&d_u, local_grid_size));
    checkHipErrors(hipMalloc(&d_unew, local_grid_size));
    checkHipErrors(hipMalloc(&d_f, local_grid_size));
    checkHipErrors(hipMalloc(&d_u_exact, local_grid_size));

    double *h_send_buffer_top = nullptr, *h_send_buffer_bottom = nullptr;
    double *h_recv_buffer_top = nullptr, *h_recv_buffer_bottom = nullptr;
    size_t row_size = N * sizeof(double);
    
    checkHipErrors(hipHostMalloc(&h_send_buffer_top, row_size));
    checkHipErrors(hipHostMalloc(&h_send_buffer_bottom, row_size));
    checkHipErrors(hipHostMalloc(&h_recv_buffer_top, row_size));
    checkHipErrors(hipHostMalloc(&h_recv_buffer_bottom, row_size));

    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim((N + TILE_X - 1) / (TILE_X), (local_padded_rows + TILE_Y - 1)/ (TILE_Y));
    int num_blocks = gridDim.x * gridDim.y;
    
    double *d_partial_max = nullptr, *d_tmp1 = nullptr;
    checkHipErrors(hipMalloc((void**)&d_partial_max, num_blocks * sizeof(double)));
    checkHipErrors(hipMalloc((void**)&d_tmp1, num_blocks * sizeof(double)));
    double h_local_max = 0.0;

    size_t sh_jacobi_bytes = (TILE_X+2)*(TILE_Y+2) * sizeof(double);
    size_t sh_check_bytes = (TILE_X * TILE_Y) * sizeof(double);

    init_local_grids<<<gridDim, blockDim>>>(d_u, d_unew, d_f, d_u_exact, N, h, local_N, global_start_row);
    checkHipErrors(hipGetLastError());
    checkHipErrors(hipDeviceSynchronize());

    int iter = 0;
    double global_max_diff = 0.0;
    MPI_Request requests[4];
    MPI_Status statuses[4];

    // --- Performance Tracking (Kernel Timers) ---
    hipEvent_t k_start, k_stop;
    checkHipErrors(hipEventCreate(&k_start));
    checkHipErrors(hipEventCreate(&k_stop));
    float total_update_ms = 0.0f;
    float total_conv_ms = 0.0f;

    double start_time = MPI_Wtime();

    for (iter = 0; iter < MAX_ITER; ++iter) {
        
        MPI_Irecv(h_recv_buffer_top, N, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(h_recv_buffer_bottom, N, MPI_DOUBLE, rank_down, 1, MPI_COMM_WORLD, &requests[1]);

        checkHipErrors(hipMemcpy(h_send_buffer_top, d_u + (1 * N), row_size, hipMemcpyDeviceToHost));
        checkHipErrors(hipMemcpy(h_send_buffer_bottom, d_u + (local_N * N), row_size, hipMemcpyDeviceToHost));

        MPI_Isend(h_send_buffer_top, N, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(h_send_buffer_bottom, N, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, &requests[3]);

        MPI_Waitall(4, requests, statuses);

        if (rank_up != MPI_PROC_NULL) {
            checkHipErrors(hipMemcpy(d_u + (0 * N), h_recv_buffer_top, row_size, hipMemcpyHostToDevice));
        }
        if (rank_down != MPI_PROC_NULL) {
            checkHipErrors(hipMemcpy(d_u + ((local_N+1) * N), h_recv_buffer_bottom, row_size, hipMemcpyHostToDevice));
        }

        // Measure Update Kernel (Compute only)
        checkHipErrors(hipEventRecord(k_start));
        jacobi_mpi<<<gridDim, blockDim, sh_jacobi_bytes>>>(d_u, d_unew, d_f, d_partial_max, N, h, local_N, global_start_row);
        checkHipErrors(hipEventRecord(k_stop));
        checkHipErrors(hipEventSynchronize(k_stop));
        float dt = 0.0f;
        checkHipErrors(hipEventElapsedTime(&dt, k_start, k_stop));
        total_update_ms += dt;
        
        checkHipErrors(hipGetLastError());

        // Measure Convergence Reduction Kernel (Device part only)
        checkHipErrors(hipEventRecord(k_start));
        double local_max_diff = reduce_final_max(d_partial_max, num_blocks, d_tmp1, &h_local_max);
        checkHipErrors(hipEventRecord(k_stop));
        checkHipErrors(hipEventSynchronize(k_stop));
        checkHipErrors(hipEventElapsedTime(&dt, k_start, k_stop));
        total_conv_ms += dt;

        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        std::swap(d_u, d_unew);

        if (rank == 0 && iter % PRINT_FREQ == 0) {
            cout << "Iter: " << setw(5) << iter << ", Max Diff: " << scientific << global_max_diff << endl;
        }

        if (global_max_diff < TOLERANCE) {
            if (rank == 0) {
                cout << "Convergence reached at iteration " << iter << "!" << endl;
                cout<< "Final Max Diff: " << scientific << global_max_diff << endl;
            }
            break;
        }
    }

    double end_time = MPI_Wtime();

    if (iter == MAX_ITER) {
        if (rank == 0) {
            cout << "Solver stopped after reached max iterations (" << MAX_ITER << ")." << endl;
            cout << "Final Max Diff: " << scientific << global_max_diff << endl;
        }
    }
    if (rank == 0) {
        cout << "Total time to solution: " << fixed << setprecision(10) << (end_time - start_time) << "s" << endl;
        cout << "Iterations: " << iter << endl;
    }
    
    check_soln_mpi<<<gridDim, blockDim, sh_check_bytes>>>(d_u, d_u_exact, d_partial_max, N, local_N);
    checkHipErrors(hipGetLastError());

    double local_max_error = reduce_final_max(d_partial_max, num_blocks, d_tmp1, &h_local_max);
    double global_max_error = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << fixed << setprecision(10);
        cout << "Maximum Absolute Error: " << global_max_error << endl;
        if (global_max_error > 1.0e-4) {
            cout << "Warning: Error is larger than expected." << endl;
        } else {
            cout << "Result is correct within expected numerical error." << endl;
        }

        // --- Calculate Roofline Metrics based on Local Rank Performance ---
        
        // Update Loop: 9 FLOPs/site, 24 Bytes/site
        // Note: Using Rank 0's local workload as representative for the system rate per GPU
        double local_flops_update = (double)local_N * N * 9.0 * iter;
        double local_bytes_update = (double)local_N * N * 24.0 * iter;
        double update_seconds = total_update_ms / 1000.0;
        
        double tflops_update = (local_flops_update / update_seconds) / 1.0e12;
        double ai_update = tflops_update / (local_bytes_update / update_seconds);

        // Convergence Loop: Approx 1 FLOP/block, 8 Bytes/block
        double local_flops_conv = (double)num_blocks * 1.0 * iter;
        double conv_seconds = total_conv_ms / 1000.0;
        
        double ai_conv = 0.125; // 1 op / 8 bytes
        double tflops_conv = (local_flops_conv / conv_seconds) / 1.0e12;

        cout << "loop_name,AI,TFLOPS" << endl;
        cout << "update_loop," << ai_update << "," << tflops_update << endl;
        cout << "convergence_loop," << ai_conv << "," << tflops_conv << endl;
    }

    checkHipErrors(hipFree(d_u));
    checkHipErrors(hipFree(d_unew));
    checkHipErrors(hipFree(d_f));
    checkHipErrors(hipFree(d_u_exact));
    checkHipErrors(hipFree(d_partial_max)); 
    checkHipErrors(hipFree(d_tmp1));

    checkHipErrors(hipHostFree(h_send_buffer_top));
    checkHipErrors(hipHostFree(h_send_buffer_bottom));
    checkHipErrors(hipHostFree(h_recv_buffer_top));
    checkHipErrors(hipHostFree(h_recv_buffer_bottom));
    
    checkHipErrors(hipEventDestroy(k_start));
    checkHipErrors(hipEventDestroy(k_stop));

    MPI_Finalize();
    return 0;
}