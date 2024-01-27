#include "utils.h"
#include "cuda_core.cuh"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("usage: ./matmul1 m n k\n");
        return 0;
    } 

    std::srand(std::time(nullptr));

    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);

    half* h_A = random_data(m * k);
    half* h_B = random_data(k * n);
    half* h_C = empty_data(m * n);

    cuda_core(m, n, k, h_A, h_B, h_C);
    return 0;
}