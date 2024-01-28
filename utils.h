#pragma once 

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include "cuda_fp16.h"

half* random_data(int size);

half* empty_data(int size);

half* copy_data(half* data, int size);

void transpose(half* matrix, int m, int n);

void check(half* A, half* B, int size);

#define ROUND(x, y) (((x) + (y) - 1) / (y))
