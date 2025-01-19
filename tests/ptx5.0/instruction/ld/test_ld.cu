__constant__ int const_data[4];

__global__ void test_ld_const(int *output) {
    int idx = threadIdx.x;
    if (idx < 4) {
        output[idx] = const_data[idx];
    }
}


