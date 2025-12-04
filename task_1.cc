#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <hipcub/hipcub.hpp>

using namespace std;

#define checkHipErrors(val) check_hip( (val), #val, __FILE__, __LINE__ )
inline void check_hip(hipError_t result, const char *const func, const char *const file, const int line) {
    if (result != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d, %s at '%s'\n", file, line, hipGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// tile dims -> each block contains TILE_X x TILE_Y threads
#ifndef TILE_X
#define TILE_X 16
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif 

/**
 * @brief initialize grids (u, unew, f, u_exact) on device
 */
__global__ void init_grids(double* __restrict__ u, double* __restrict__ unew, double* __restrict__ f, double* __restrict__ u_exact, 
                            int N, double h) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // check bounds (grid may not be perfectly divisible by block size)
    if (i < N && j < N) {
        double x = i * h;
        double y = j * h;
        int idx = i * N + j;

        u_exact[idx] = sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
        f[idx] = -8.0 * M_PI * M_PI * sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);

        u[idx] = 0.0;
        unew[idx] = 0.0;
    }
}

/**
 * @brief Kernel for one Jacobi iteration + local block max diff.
 */
__global__ void jacobi(const double* __restrict__ u, double* __restrict__ unew,
                                     const double* __restrict__ f, double* __restrict__ d_partial_max, int N, double h) {
    // global thread idx (block_start + local_thread_idx)
    int j = blockIdx.x * TILE_X + threadIdx.x;
    int i = blockIdx.y * TILE_Y + threadIdx.y;
    int idx = i * N + j;

    // smem: (TILE_Y + 2) x (TILE_X + 2) -> halo for buffer when computing jacobi
    extern __shared__ double smem[];
    int scols = TILE_X + 2;
    // int srows = TILE_Y + 2; 

    double* s_tile = smem; //ptr to tile data; srows * scols

    int total_cells = (TILE_Y + 2) * (TILE_X + 2); 
    int nthreads = TILE_X * TILE_Y;
    
    // load data into smem
    for (int k=0; k<total_cells; k +=nthreads) {
        int s_i = (threadIdx.y * TILE_X + threadIdx.x) + k;
        if (s_i < total_cells) {
            // -1 to convert from smem with halo to hbm w/o halo
            int global_i = blockIdx.y * TILE_Y  + (s_i / scols)-1;
            int global_j = blockIdx.x * TILE_X + (s_i % scols)-1;

            double val = 0.0; 
            // loading in bounds, too, for calculations
            if (global_i >= 0 && global_i < N && global_j >= 0 && global_j < N) {
                val = u[global_i * N + global_j]; // load from hbm
            }
            s_tile[s_i] = val;
        }
    }
    __syncthreads(); // finish loading data

    // compute jacobi for interior points. note N is size of entire grid
    double threaddiff = 0.0;
    int s_i = threadIdx.y + 1; // +1 cuz halo in smem
    int s_j = threadIdx.x + 1;
    if (i>0 && i<N-1 && j>0 && j<N-1) {
        // val avoids hbm read
        double up = s_tile[(s_i - 1) * scols + s_j];
        double down = s_tile[(s_i+1)*scols+s_j];
        double left = s_tile[s_i * scols + (s_j-1)];
        double right = s_tile[s_i * scols + (s_j+1)];
        double val = 0.25 * (up + down + left+ right - h * h * f[idx]);
        unew[idx] = val;
        threaddiff = fabs(val - s_tile[s_i * scols + s_j]);
    }

    // Use compile-time constant (macros) for template argument
    using BlockReduce = hipcub::BlockReduce<double, TILE_X * TILE_Y>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // FIX: Removed <double> from hipcub::Max
    double block_max = BlockReduce(temp_storage).Reduce(threaddiff, hipcub::Max());

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        d_partial_max[blockIdx.y * gridDim.x + blockIdx.x] = block_max;
    }
}

/**
 * @brief Kernel to check computed soln against exact soln
 */
__global__ void check_soln (const double* __restrict__ u, const double* __restrict__ u_exact,
                           double* __restrict__ d_partial_max_error, const int N) {
    // shared mem for block reduction
    extern __shared__ double s_data[];

    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    unsigned int block_size = blockDim.x * blockDim.y;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    double err = 0.0;
    // check all points (in case boundary was altered)
    if (i < N && j < N) {
        int idx = i * N + j;
        err = fabs(u_exact[idx] - u[idx]);
    }
    s_data[tid] = err;
    __syncthreads(); // calc all errs before reducing

    // reduce in block
    for (unsigned int s = block_size / 2; s>0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmax(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    // write results to grid array
    if (tid == 0) {
        int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        d_partial_max_error[block_idx] = s_data[0];
    }
}

/**
 * @brief Manual parallel reduction kernel 
 */
#define FINAL_RED_BLOCK_SZ 256
__global__ void block_reduce(const double* __restrict__ d_partial, double* __restrict__ d_out, int M) {
    __shared__ double sdata[FINAL_RED_BLOCK_SZ];
    unsigned int tid = threadIdx.x; // 1d block
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid; // can be launched w/ multiple blocks
    double tmax = 0.0;
    
    // load in 2 elem from HBM at once 
    if (i < (unsigned int)M) tmax = d_partial[i];
    if (i + blockDim.x < (unsigned int)M) {
        double v = d_partial[i + blockDim.x];
        if (v > tmax) tmax = v;
    }
    sdata[tid] = tmax;
    __syncthreads(); // sync after loading in data

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// host helper. compute final max by launching block_reduc iteratively 
double reduce_final_max(double* d_partial, int num_partials, double* d_tmp1) {
    int M = num_partials; 
    int threads = FINAL_RED_BLOCK_SZ;

    // ptrs for ping-pong buffering
    double* d_in = d_partial;
    double* d_out = d_tmp1;

    do {
        int blocks = (M + threads*2 - 1) / (threads*2);
        block_reduce<<<blocks, threads>>>(d_in, d_out, M);
        checkHipErrors(hipGetLastError());

        std::swap(d_in, d_out); 

        M = blocks; // each block reduces to 1 elem
    } while (M > 1);

    double res = 0.0;
    // expensive
    checkHipErrors(hipMemcpy(&res, d_in, sizeof(double), hipMemcpyDeviceToHost)); // read from d_in bc std::swap
    return res;
}

/**
 * @brief run Jacobi iterative solver on GPU!
 */
int main(int argc, char** argv) {
    // 1. constant setup + alloc dev mem
    int N = 201;
    if (argc >= 2) N = atoi(argv[1]);
    
    int MAX_ITER = 100000;
    if (argc >= 3) MAX_ITER = atoi(argv[2]);
    const double h = 1.0 / (N-1);
    const double TOLERANCE = 1.0e-8;
    const int PRINT_FREQ = 500;

    cout << "Running Jacobi solver for N = " << N << " x " << N << " for " << MAX_ITER << " iterations" << endl;

    // allocate gpu mem
    double *d_u, *d_unew, *d_f, *d_u_exact;
    size_t size = N*N*sizeof(double);
    checkHipErrors(hipMalloc(&d_u, size));
    checkHipErrors(hipMalloc(&d_unew, size));
    checkHipErrors(hipMalloc(&d_f, size));
    checkHipErrors(hipMalloc(&d_u_exact, size));

    // 2. grid / block setup
    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim((N + TILE_X - 1) / (TILE_X), (N + TILE_Y - 1)/ (TILE_Y));
    int num_blocks = gridDim.x * gridDim.y;

    // alloc gpu mem for partial reducs
    double *d_partial_max;
    checkHipErrors(hipMalloc((void**)&d_partial_max, num_blocks * sizeof(double)));

    // alloc gpu mem for single final reduc result
    double* d_tmp1;
    checkHipErrors(hipMalloc((void**)&d_tmp1, num_blocks * sizeof(double)));

    // alloc shared mem for check_soln kernel PER THREAD BLOCK 
    size_t sh_check_bytes = (TILE_X * TILE_Y) * sizeof(double);

    // alloc shared mem for jacobi kernel: tile. amt of shared mem PER THREAD BLOCK
    size_t sh_bytes = (TILE_X+2)*(TILE_Y+2) * sizeof(double);

    // 3. init grids on GPU
    init_grids<<<gridDim, blockDim>>>(d_u, d_unew, d_f, d_u_exact, N, h);
    checkHipErrors(hipGetLastError());
    checkHipErrors(hipDeviceSynchronize()); // wait for init to finish

    // 4. jacobi iterative solver
    int iter = 0;
    double final_max = 0.0;

    // timing events (Overall)
    hipEvent_t start, stop;
    checkHipErrors(hipEventCreate(&start));
    checkHipErrors(hipEventCreate(&stop));

    // timing events (Per-Kernel Analysis for Roofline)
    hipEvent_t k_start, k_stop;
    checkHipErrors(hipEventCreate(&k_start));
    checkHipErrors(hipEventCreate(&k_stop));
    float total_update_ms = 0.0f;
    float total_conv_ms = 0.0f;

    cout << "Starting Jacobi solver on GPU..." << endl;
    checkHipErrors(hipEventRecord(start));

    
    for (iter = 0; iter < MAX_ITER; ++iter) {
        
        // --- 1. Update Step (Jacobi Kernel) ---
        checkHipErrors(hipEventRecord(k_start));
        jacobi<<<gridDim, blockDim, sh_bytes>>>(d_u, d_unew, d_f, d_partial_max, N, h);
        checkHipErrors(hipEventRecord(k_stop));
        checkHipErrors(hipEventSynchronize(k_stop));
        float dt = 0.0f;
        checkHipErrors(hipEventElapsedTime(&dt, k_start, k_stop));
        total_update_ms += dt;
        
        checkHipErrors(hipGetLastError());

        // --- 2. Convergence Step (Reduction) ---
        checkHipErrors(hipEventRecord(k_start));
        final_max = reduce_final_max(d_partial_max, num_blocks, d_tmp1);
        checkHipErrors(hipEventRecord(k_stop));
        checkHipErrors(hipEventSynchronize(k_stop));
        checkHipErrors(hipEventElapsedTime(&dt, k_start, k_stop));
        total_conv_ms += dt;

        swap(d_u, d_unew);

        if (iter % PRINT_FREQ == 0) {
            cout << "Iter: " << setw(5) << iter << ", Max Diff: " << scientific << final_max << endl;
        }

        if (final_max < TOLERANCE) {
            cout << "Convergence reached at iteration " << iter << "!" << endl;
            cout<< "Final Max Diff: " << scientific << final_max << endl;
            break;
        }
    }

    // wait for gpu work to stop before timing (Total Wall Time)
    checkHipErrors(hipEventRecord(stop));
    checkHipErrors(hipEventSynchronize(stop)); 
    float ms = 0.0f;
    checkHipErrors(hipEventElapsedTime(&ms, start, stop));

   
    if (iter == MAX_ITER) {
        cout << "Solver stopped after reached max iterations (" << MAX_ITER << ")." << endl;
        cout << "Final Max Diff: " << scientific << final_max << endl;
    }
    cout << "Total time to solution: " << fixed << setprecision(10) << (ms / 1000.0) << "s" << endl;
    
    // 5. check soln correctness
    double *d_partial_max_err = d_partial_max;
    check_soln<<<gridDim, blockDim, sh_check_bytes>>>(d_u, d_u_exact, d_partial_max_err, N);
    checkHipErrors(hipGetLastError());

    // re-use high-perf reduction!
    double max_err = reduce_final_max(d_partial_max_err, num_blocks, d_tmp1);

    cout << fixed << setprecision(10);
    cout << "Maximum Absolute Error: " << max_err << endl;

    if (max_err > 1.0e-4) {
        cout << "Warning: Error is larger than expected." << endl;
    } else {
        cout << "Result is correct within expected numerical error." << endl;
    }

    // --- Performance Metrics for Roofline ---

    // UPDATE LOOP METRICS
    // FLOPs: 7 stencil + 2 residual = 9 per site
    double total_flops_update = (double)N * N * 9.0 * iter;
    // Bytes: Read u (8), Read f (8), Write unew (8) = 24 per site
    double total_bytes_update = (double)N * N * 24.0 * iter;
    // Use the ACCUMULATED KERNEL TIME (total_update_ms) for accuracy
    double update_seconds = total_update_ms / 1000.0;
    double tflops_update = (total_flops_update / update_seconds) ;
    double ai_update = total_flops_update / total_bytes_update; 

    // CONVERGENCE LOOP METRICS (Reduction)
    // Reduce processes 'num_blocks' elements. 
    // FLOPs: approx 1 comparison per element.
    // Bytes: approx 1 read (8 bytes) per element. (Strictly read+write for multi-pass, but read dominant).
    // Note: Reduction is latency bound on small grids, so TFLOPS will be very low.
    double total_flops_conv = (double)num_blocks * 1.0 * iter;
    double total_bytes_conv = (double)num_blocks * 8.0 * iter;
    double conv_seconds = total_conv_ms / 1000.0;
    
    // Theoretical AI for simple max reduction: 1 op / 8 bytes = 0.125
    double ai_conv = 0.125; 
    double tflops_conv = (total_flops_conv / conv_seconds) / 1.0e12;

    cout << "loop_name,AI,TFLOPS" << endl;
    cout << "update_loop," << ai_update << "," << tflops_update << endl;
    cout << "convergence_loop," << ai_conv << "," << tflops_conv << endl;

    // 6. mem cleanup
    checkHipErrors(hipFree(d_u));
    checkHipErrors(hipFree(d_unew));
    checkHipErrors(hipFree(d_f));
    checkHipErrors(hipFree(d_u_exact));
    checkHipErrors(hipFree(d_partial_max)); 
    checkHipErrors(hipFree(d_tmp1));
    checkHipErrors(hipEventDestroy(start));
    checkHipErrors(hipEventDestroy(stop));
    checkHipErrors(hipEventDestroy(k_start));
    checkHipErrors(hipEventDestroy(k_stop));

    return 0;
}