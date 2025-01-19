/* This file will test PTX instructions cvta, ld, isspacep when using generic address in const state space */

// allocate const memory
__constant__ int const_data[4] = {1, 2, 3, 4};

__global__ void test_ispacep_of_const_generic(int* d) {
    // assuming all block and grid are 1D
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < 4) {
        int* addr = const_data + index;
        int result = index;
        asm volatile(
        ".reg .pred %p;\n\t"
        "isspacep.const %p, %1;\n\t" 
        "@%p mov.s32 %0, 1;\n\t"
        : "=r"(result)  
        : "l"(addr)
        );
        d[index] = result;     
    }
}

__global__ void test_cvta_to_generic(int* d) {
    // assuming all block and grid are 1D
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < 4) {

        // PTX will automatically convert const address to generic address when calculate the address as C++ pointer
        int* generic_addr = &const_data[index];
        
        int result = (long) generic_addr;

        // we use isspacep to check if the address is valid
        asm volatile(
        ".reg .pred %p;\n\t"
        "isspacep.const %p, %1;\n\t" 
        "@%p mov.s32 %0, 1;\n\t"
        : "=r"(result)  
        : "l"(generic_addr)
        );
        
        d[index] = result;
    }
}

__global__ void test_ld_from_const(int* d) {
    // assuming all block and grid are 1D
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < 4) {
        d[index] = const_data[index];
    }
}

__global__ void test_ld_from_const_generic(int* d) {
    // assuming all block and grid are 1D
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < 4) {

        // PTX will automatically convert const address to generic address when calculate the address as C++ pointer
        int* generic_addr = const_data;

        generic_addr += index;
        
        int result = (long) generic_addr;

        // we use ld from generic to check correctness
        asm volatile(
        "ld.u32 %0, [%1];\n\t" 
        : "=r"(result)  
        : "l"(generic_addr)
        );
        
        d[index] = result;
    }
}

__global__ void test_cvta_from_generic(int* d) {
    // assuming all block and grid are 1D
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < 4) {

        // PTX will automatically convert const address to generic address when calculate the address as C++ pointer
        int* generic_addr = const_data;

        generic_addr += index;
        
        int result = (long) generic_addr;

        // convert generic address back and ld from const state space
        asm volatile(
        ".reg .u64 t1;\n\t"
        "cvta.to.const.u64 t1, %1;\n\t" 
        "ld.const.u32 %0, [t1];\n\t"
        : "=r"(result)  
        : "l"(generic_addr)
        );
        
        d[index] = result;
    }
}



