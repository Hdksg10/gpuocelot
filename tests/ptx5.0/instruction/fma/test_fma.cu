#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

/* Macros to allow half & half2 to be used by inline assembly.
 * Copy from cuda_fp16.hpp
 */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

__global__ void test_fmaf16(half* d, half* a, half* b, half* c) {
    half val; 
    asm( "{fma.rn.f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(*a)),"h"(__HALF_TO_CUS(*b)), "h"(__HALF_TO_US(*c))); \
    *d = val;
}


__global__ void test_fmaf16_sat(half* d) {
    short a_val = 0x03ff;
    short b_val = 0x3c00;
    short c_val = 0x3c00;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    half c = *reinterpret_cast<half*>(&c_val);
    half val; 
    asm( "{fma.rn.sat.f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b)), "h"(__HALF_TO_US(c))); \
    *d = val;
}

__global__ void test_fmaf16_ftz(half* d) {
    short a_val = 0x03ff; // the largest subnormal number
    half a = *reinterpret_cast<half*>(&a_val);
    half b = a;
    half c = a;
    half val; 
    asm( "{fma.rn.ftz.f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b)), "h"(__HALF_TO_US(c))); \
    *d = val;
}

__global__ void test_fmaf16_ftz_sat(half* d) {
    short a_val = 0x03ff;
    short b_val = 0x3c00;
    short c_val = 0x3c0a;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    half c = *reinterpret_cast<half*>(&c_val);
    half val; 
    asm( "{fma.rn.ftz.sat.f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b)), "h"(__HALF_TO_US(c))); \
    *d = val;
}

__global__ void test_fmaf16x2(half2* d, half2* a, half2* b, half2* c) {
    half2 val; 
    asm( "{fma.rn.f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(*a)),"r"(__HALF2_TO_CUI(*b)), "r"(__HALF2_TO_CUI(*c))); \
    *d = val;
}

__global__ void test_fmaf16x2_sat(half2* d) {
    half2 val; 
    short ah_val = 0x03ff;
    short al_val = 0x3c00;
    short bh_val = 0x3c00;
    short bl_val = 0x03ff;
    short ch_val = 0x3c00;
    short cl_val = 0xbc01;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half ch = *reinterpret_cast<half*>(&ch_val);
    half cl = *reinterpret_cast<half*>(&cl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    half2 c(ch, cl);
    asm( "{fma.rn.sat.f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b)), "r"(__HALF2_TO_CUI(c))); \
    *d = val;
}

__global__ void test_fmaf16x2_ftz(half2* d) {
    half2 val; 
    short ah_val = 0x03ff; // the largest subnormal number
    short al_val = 0x83ff;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half2 a(ah,al);
    asm( "{fma.rn.ftz.f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(a)), "r"(__HALF2_TO_CUI(a))); \
    *d = val;
}

__global__ void test_fmaf16x2_ftz_sat(half2* d) {
    half2 val; 
    short ah_val = 0x03ff;
    short al_val = 0x3c00;
    short bh_val = 0x3c00;
    short bl_val = 0x840a;
    short ch_val = 0x3c00;
    short cl_val = 0x3c00;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half ch = *reinterpret_cast<half*>(&ch_val);
    half cl = *reinterpret_cast<half*>(&cl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    half2 c(ch, cl);
    asm( "{fma.rn.ftz.sat.f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b)), "r"(__HALF2_TO_CUI(c))); \
    *d = val;
}