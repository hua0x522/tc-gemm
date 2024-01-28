#include "utils.h"

half* random_data(int size) {
    half* handle = (half*)malloc(size * sizeof(half));
    for (int i = 0; i < size; i++) {
        handle[i] = (1.0 * (rand() % 10000)) / 100.0;
    }
    return handle;
}

half* empty_data(int size) {
    half* handle = (half*)malloc(size * sizeof(half));
    for (int i = 0; i < size; i++) {
        handle[i] = 0;
    }
    return handle;
}

half* copy_data(half* data, int size) {
    half* handle = (half*)malloc(size * sizeof(half));
    for (int i = 0; i < size; i++) {
        handle[i] = data[i];
    }
    return handle;
}

void transpose(half* matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            half temp = matrix[i * n + j];
            matrix[i * n + j] = matrix[j * n + i];
            matrix[j * n + i] = temp;
        }
    }
}

void check(half* A, half* B, int size) {
    for (int i = 0; i < size; i++) {
        float a = __half2float(A[i]);
        float b = __half2float(B[i]);
        if (fabs(a - b) / a >= 1e-3) {
            printf("error at %d, %lf, %lf\n", i, a, b);
            return;
        }
    }
    printf("check success\n");
}