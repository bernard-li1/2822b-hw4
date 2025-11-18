#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cub/cub.cuh>

using namespace std;
//TODO: implement 2D tiling in jacobi function (each thread can load in 2D)

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, const char *const func, const char *const file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d, %s at '%s'\n", file, line, cudaGetErrorString(result), func);
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
 *
 * Sets up forcing function f, exact solution u_exact, init state of u and unew
 * Applies Dirichlet BCs to u and unew based on exact solution (zero boundaries).
 */
// restrict: any data written to ptr is not read by any other ptr w/ restrict
// avoids unnecessary compiler reads (alias checks). on H100, allows register caching + other ptr aliasing optimizations
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
 * @brief Kernel for one Jacobi iteration + local block max diff. (ideally) 2D tiling using smem.
 * Each thread block loads/computes on TILE_X * TILE_Y vals; 1:1 thread:elem
 * Computes unew from u for all interior points
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
    int srows = TILE_Y + 2;

    double* s_tile = smem; //ptr to tile data; srows * scols

    int total_cells = srows * scols; 
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

    // blockreduce
    using BlockReduce = cub::BlockReduce<double, nthreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // returns max for thread (0,0), other threads get undefined val
    double block_max = BlockReduce(temp_storage).Reduce(threaddiff, cub::Max<double>());


    if (threadIdx.y == 0 && threadIdx.x == 0) {
        d_partial_max[blockIdx.y * gridDim.x + blockIdx.x] = block_max;
    }
}

/**
 * @brief Kernel to check computed soln against exact soln
 * First stage of two-stage reduction (1st: red in block; 2nd: red in grid).
 * Block finds local max err & writes to d_partial_max_error
 */
__global__ void check_soln (const double* __restrict__ u, const double* __restrict__ u_exact,
                           double* __restrict__ d_partial_max_error, const int N) {
    // shared mem for block reduction
    extern __shared__ double s_data[];
    // avoid hard-coding smem size -- would need to launch max needed size even if at runtime we using less threads

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // in-block id
    int block_size = blockDim.x * blockDim.y;

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

// can use blockReduce instead of the below. reduce_final_max is a parallel block red.
/**
 * @brief Manual parallel reduction kernel -> reduce partial block maxima to single val w parallel blocks
 * 
 * In reduce_final_max, launches multiple parallel blocks if need be, instead of just 1 block. (GPU is idling waiting on this kernel at this pt.)
 */
#define FINAL_RED_BLOCK_SZ 256
__global__ void block_reduce(const double* __restrict__ d_partial, double* __restrict__ d_out, int M) {
    __shared__ double sdata[FINAL_RED_BLOCK_SZ];
    unsigned int tid = threadIdx.x; // 1d block
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid; // can be launched w/ multiple blocks. global i
    double tmax = 0.0;
    // load in 2 elem from HBM at once (like 'grid-stride looping' but not really)
    if (i < M) tmax = d_partial[i];
    if (i + blockDim.x < M) {
        double v = d_partial[i + blockDim.x];
        if (v > tmax) tmax = v;
    }
    sdata[tid] = tmax;
    __syncthreads(); // sync after loading in data

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}
// safer to allocate 2 temp buffers. leaves d_partial_max untouched.
// host helper. compute final max by launching block_reduc iteratively (multiple blocks if enough elem to reduce)
double reduce_final_max(double* d_partial, int num_partials, double* d_tmp1) {
    int M = num_partials; // num elem to reduce total; len(*d_partial)
    int threads = FINAL_RED_BLOCK_SZ;

    // ptrs for ping-pong buffering
    double* d_in = d_partial;
    double* d_out = d_tmp1;

    do {
        int blocks = (M + threads*2 - 1) / (threads*2);
        block_reduce<<blocks, threads>>(d_in, d_out, M);
        checkCudaErrors(cudaGetLastError());

        std::swap(d_in, d_out); // overwrites d_partial, but this is fine since we don't use it afterwards

        M = blocks; // each block reduces to 1 elem
    } while (M > 1);

    double res = 0.0;
    // expensive
    checkCudaErrors(cudaMemcpy(&res, d_in, sizeof(double), cudaMemcpyDeviceToHost)); // read from d_in bc std::swap
    return res;

}

// THE BELOW KERNEL LAUNCHES 1 BLOCK TO REDUCE. HOWEVER, GRID STRIDE INEFFICIENT IF NUM ELEM >> THREADS IN BLOCK. 
// THE ABOVE REDUC LAUNCHES MULTIPLE KERNELS TO REDUCE, THEN REDUCES THAT. IT ALSO LAUNCHES JUST 1 KERNEL IF NECESSARY. STRICTLY BETTER PERF.
// the below function can be replaced with cub::BlockReduce.
// BlockReduce uses warp-based reduction & only 1 __syncthreads()
/**
 * @brief reduces partial results from first-stage kernels
 * calculates reduction across grid.
 * we fix block size for kernel; must be power of 2.
 * 
 * run with single block.
 */
// #define FINAL_RED_BLOCK_SZ 256
// __global__ void final_reduc(const double* d_partial_results, double* d_final_result, int N_partial) {
//     __shared__ double s_data[FINAL_RED_BLOCK_SZ];
//     int tid = threadIdx.x; // 1d thread block

//     // grid-stride loop. each threads gets max val for owned elem.
//     double tmax = 0.0;
//     for (unsigned int i = tid; i < N_partial; i += FINAL_RED_BLOCK_SZ) {
//         tmax = fmax(tmax, d_partial_results[i]);
//     }
//     s_data[tid] = tmax;
//     __syncthreads(); // wait for all threads to compete red

//     // use BlockReduce for non-naive version
//     for (unsigned int s = FINAL_RED_BLOCK_SZ / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             s_data[tid] = fmax(s_data[tid], s_data[tid + s]);
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         *d_final_result = s_data[0];
//     }
// }

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
    checkCudaErrors(cudaMalloc(&d_u, size));
    checkCudaErrors(cudaMalloc(&d_unew, size));
    checkCudaErrors(cudaMalloc(&d_f, size));
    checkCudaErrors(cudaMalloc(&d_u_exact, size));

    // 2. grid / block setup
    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim((N + TILE_X - 1) / (TILE_X), (N + TILE_Y - 1)/ (TILE_Y));
    int num_blocks = gridDim.x * gridDim.y;

    // alloc gpu mem for partial reducs
    double *d_partial_max;
    checkCudaErrors(cudaMalloc((void**)&d_partial_max, num_blocks * sizeof(double)));

    // alloc gpu mem for single final reduc result
    double* d_tmp1;
    checkCudaErrors(cudaMalloc((void**)&d_tmp1, num_blocks * sizeof(double)));

    // alloc shared mem for check_soln kernel PER THREAD BLOCK (defined extern so we can easily change tile_x and tile_y vals)
    size_t sh_check_bytes = (TILE_X * TILE_Y) * sizeof(double);

    // alloc shared mem for jacobi kernel: tile. amt of shared mem PER THREAD BLOCK
    size_t sh_bytes = (TILE_X+2)*(TILE_Y+2) * sizeof(double);

    // 3. init grids on GPU
    init_grids<<gridDim, blockDim>>(d_u, d_unew, d_f, d_u_exact, N, h);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize()); // wait for init to finish

    // 4. jacobi iterative solver
    int iter = 0;
    double final_max = 0.0;

    // timing events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    cout << "Starting Jacobi solver on GPU..." << endl;
    checkCudaErrors(cudaEventRecord(start));

    
    for (iter = 0; iter < MAX_ITER; ++iter) {
        // compute unew from u; compute maxdiff
        jacobi<<gridDim, blockDim, sh_bytes>>(d_u, d_unew, d_f, d_partial_max, N, h);
        checkCudaErrors(cudaGetLastError());

        // reduce partials to final max on device
        final_max = reduce_final_max(d_partial_max, num_blocks, d_tmp1);

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

    // wait for gpu work to stop before timing
    checkCudaErrors(cudaEventRecord(stop));
    // device driver timing
    checkCudaErrors(cudaEventSynchronize(stop)); // Wait until the completion of all device work preceding the most recent call to cudaEventRecord() ; cudaDevSync is not necessary
    float ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));

   
    if (iter == MAX_ITER) {
        cout << "Solver stopped after reached max iterations (" << MAX_ITER << ")." << endl;
        cout << "Final Max Diff: " << scientific << final_max << endl;
    }
    cout << "Total time to solution: " << fixed << setprecision(10) << (ms / 1000.0) << "s" << endl;
    
    // 5. check soln correctness
    // d_u is most recent grid after swap. reusing partial diff mem
    double *d_partial_max_err = d_partial_max;
    check_soln<<gridDim, blockDim, sh_check_bytes>>(d_u, d_u_exact, d_partial_max_err, N);
    checkCudaErrors(cudaGetLastError());

    // re-use high-perf reduction!
    double max_err = reduce_final_max(d_partial_max_err, num_blocks, d_tmp1);

    cout << fixed << setprecision(10);
    cout << "Maximum Absolute Error: " << max_err << endl;

    if (max_err > 1.0e-4) {
        cout << "Warning: Error is larger than expected." << endl;
    } else {
        cout << "Result is correct within expected numerical error." << endl;
    }

    // 6. mem cleanup
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_unew));
    checkCudaErrors(cudaFree(d_f));
    checkCudaErrors(cudaFree(d_u_exact));
    checkCudaErrors(cudaFree(d_partial_max)); 
    checkCudaErrors(cudaFree(d_tmp1));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}