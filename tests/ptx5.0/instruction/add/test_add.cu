#include <cuda_fp16.h>
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))

__global__ void test_addcc(unsigned long* d) {
    unsigned long result = 0;  

    asm(
        "add.cc.u64 %0, %1, %2;\n\t"
        "addc.u64 %0, 0, 0;\n\t"          
        : "=l"(result)                       
        : "l"(5l), "l"(0xffffffffffffffff)                              
    );

    *d = result;
}

__global__ void test_addf16(__half* d, __half* a, __half* b) {
    half result = 0.0f;
    __half val; \
    asm( "{add.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(*a)),"h"(__HALF_TO_CUS(*b))); \
    *d = val;
    // *d = __US_TO_HALF(result_us);
}

__global__ void test_subf16_v2(__half* d, __half* a, __half* b) {
    *d = __hsub(*a, *b);
}