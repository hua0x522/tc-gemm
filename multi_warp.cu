/*
This kernel is for one block contains multiply warps, and one warp process one tile(MMA tile).
*/

#include "utils.h"
#include "multi_warp.cuh"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_M 32
#define BLOCK_N 16

#define WARP_SIZE 32

__global__ void multi_warp_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    const int k_tiles = ROUND(K, MMA_K);
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int lane_id = tid % WARP_SIZE;

    const int block_row = blockIdx.x * BLOCK_M;
    const int block_col = blockIdx.y * BLOCK_N;

    __shared__ half A_smem[BLOCK_M][MMA_K];
    __shared__ half B_smem[BLOCK_N][MMA_K];
    __shared__ half C_smem[BLOCK_M][BLOCK_N];

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[4] = {0, 0, 0, 0};

    for (size_t i = 0; i < k_tiles; i++) {
        if (tid < 2 * BLOCK_M) {
            *((int4*)(&A_smem[tid / 2][0]) + tid % 2) = 
                *((int4*)(&d_A[(block_row + tid / 2) * K + i * MMA_K]) + tid % 2);
        }

        if (tid < 2 * BLOCK_N) {
            *((int4*)(&B_smem[tid / 2][0]) + tid % 2) = 
                *((int4*)(&d_B[i * MMA_K + (block_col + tid / 2) * K]) + tid % 2);
        }

        __syncthreads();
    
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[warp_m * 16 + lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[warp_n * 8 + lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3]);

        __syncthreads();
    }
    C_smem[warp_m * 16 + lane_id / 4][warp_n * 8 + (lane_id % 4) * 2] = __float2half(*((float*)&RC[0]));
    C_smem[warp_m * 16 + lane_id / 4][warp_n * 8 + (lane_id % 4) * 2 + 1] = __float2half(*((float*)&RC[1]));
    C_smem[warp_m * 16 + lane_id / 4 + 8][warp_n * 8 + (lane_id % 4) * 2] = __float2half(*((float*)&RC[2]));
    C_smem[warp_m * 16 + lane_id / 4 + 8][warp_n * 8 + (lane_id % 4) * 2 + 1] = __float2half(*((float*)&RC[3]));

    __syncthreads();

    if (tid < 2 * BLOCK_M) {
        *((int4*)(&d_C[(block_row + tid / 2) * K + block_col]) + tid % 2) = 
            *((int4*)(&C_smem[tid / 2][0]) + tid % 2);
    }
}

void multi_warp(int M, int N, int K, half* h_A, half* h_B, half* h_C) {
    transpose(h_B, K, N);

    half* d_A;
    half* d_B;
    half* d_C;
    struct timeval tv;
    double start, end;

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(half), cudaMemcpyHostToDevice);

    gettimeofday(&tv, nullptr);
    start = tv.tv_sec + tv.tv_usec / 1.0e6;

    multi_warp_kernel<<<dim3(ROUND(M, BLOCK_M), ROUND(N, BLOCK_N)), 
                            BLOCK_M * BLOCK_N * WARP_SIZE / (MMA_M * MMA_N)>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("multi-warp time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}