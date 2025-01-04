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
}


__global__ void test_addf16_sat(__half* d) {
    half result = 0.0f;
    short a_val = 0x3bff;
    short b_val = 0x3c00;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    __half val; \
    asm( "{add.sat.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_addf16_ftz(__half* d) {
    half result = 0.0f;
    short a_val = 0x03ff; // the largest subnormal number
    half a = *reinterpret_cast<half*>(&a_val);
    half b = a;
    __half val; \
    asm( "{add.ftz.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_addf16_ftz_sat(__half* d) {
    half result = 0.0f;
    short a_val = 0x03ff;
    short b_val = 0x8401;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    __half val; \
    asm( "{add.ftz.sat.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}
