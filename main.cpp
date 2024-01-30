#include "utils.h"
#include "cuda_core.cuh"
#include "base.cuh"
#include "multi_warp.cuh"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("usage: ./matmul1 m n k\n");
        return 0;
    } 

    // std::srand(std::time(nullptr));

    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);

    half* h_A = random_data(m * k);
    half* h_B = random_data(k * n);
    half* h_C = empty_data(m * n);

    half* B = copy_data(h_B, k * n);
    half* C = empty_data(m * n);

    // cuda_core(m, n, k, h_A, h_B, h_C);
    mma_base(m, n, k, h_A, h_B, h_C);
    multi_warp(m, n, k, h_A, B, C);

    check(h_C, C, m * n);
    return 0;
}