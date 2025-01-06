#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

/* Macros to allow half & half2 to be used by inline assembly.
 * Copy from cuda_fp16.hpp
 */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

__global__ void test_subcc(unsigned long* d) {
    unsigned long result = 0;  

    asm(
        "sub.cc.u64 %0, %1, %2;\n\t"
        "subc.u64 %0, 2, 0;\n\t"          
        : "=l"(result)                       
        : "l"(5l), "l"(6l)                              
    );

    *d = result;
}

__global__ void test_subf16(half* d, half* a, half* b) {
    half val; 
    asm( "{sub.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(*a)),"h"(__HALF_TO_CUS(*b))); \
    *d = val;
}


__global__ void test_subf16_sat(half* d) {
    short a_val = 0x3bff;
    short b_val = 0x3c00;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    half val; 
    asm( "{sub.sat.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_subf16_ftz(half* d) {
    short a_val = 0x03ff; // the largest subnormal number
    short b_val = 0x03fe;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    half val; 
    asm( "{sub.ftz.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_subf16_ftz_sat(half* d) {
    short a_val = 0x03fe;
    short b_val = 0x83ff;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    half val; 
    asm( "{sub.ftz.sat.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_subf16x2(half2* d, half2* a, half2* b) {
    half2 val; 
    asm( "{sub.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(*a)),"r"(__HALF2_TO_CUI(*b))); \
    *d = val;
}

__global__ void test_subf16x2_sat(half2* d) {
    half2 val; 
    short ah_val = 0x3bff;
    short al_val = 0x7bff;
    short bh_val = 0x3c00;
    short bl_val = 0x03ff;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    asm( "{sub.sat.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
    *d = val;
}

__global__ void test_subf16x2_ftz(half2* d) {
    half2 val; 
    short ah_val = 0x03ff; // the largest subnormal number
    short al_val = 0x0400;
    short bl_val = 0x03fe;
    short bh_val = 0x0401;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half2 a(ah,al);
    half2 b(bh, bl);
    asm( "{sub.ftz.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
    *d = val;
}

__global__ void test_subf16x2_ftz_sat(half2* d) {
    half2 val; 
    short ah_val = 0x03ff;
    short al_val = 0x03fe;
    short bh_val = 0x8401;
    short bl_val = 0x83ff;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    asm( "{sub.ftz.sat.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
    *d = val;
}