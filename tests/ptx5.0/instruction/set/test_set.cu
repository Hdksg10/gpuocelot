#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

/* Macros to allow half & half2 to be used by inline assembly.
 * Copy from cuda_fp16.hpp
 */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

__global__ void test_set_f16(half* d, half* a, half* b) {
    half val; 
    asm( "{set.ge.f16.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(*a)),"h"(__HALF_TO_CUS(*b))); \
    *d = val;
}

__global__ void test_set_f16_ftz(half* d) {
    half val; 
    short a_val = 0x03ff; // the largest subnormal number
    short b_val = 0x03fe;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    asm( "{set.eq.ftz.f16.f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_set_f16_boolop(half* d) {
    half val; 
    short a_val = 0x3c00;
    short b_val = 0x3c00;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    asm(".reg .pred %p;\n\t" 
        "setp.eq.s32 %p, 1, 34;\n\t"
        "set.eq.and.f16.f16 %0,%1,%2,%p;\n\t" 
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); 
    *d = val;
}

__global__ void test_set_f16_boolop_ftz(half* d) {
    half val; 
    short a_val = 0x03ff; // the largest subnormal number
    short b_val = 0x03fe;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    asm(".reg .pred %p;\n\t" 
        "setp.eq.s32 %p, 1, 34;\n\t"
        "set.eq.xor.ftz.f16.f16 %0,%1,%2,%p;\n\t" 
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); 
    *d = val;
}

__global__ void test_set_f16x2(half2* d, half2* a, half2* b) {
    half2 val;  
    asm( "{set.ge.f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(*a)),"r"(__HALF2_TO_CUI(*b))); \
    *d = val;
}

__global__ void test_set_f16x2_ftz(half2* d) {
    half2 val; 
    short ah_val = 0x03ff; // the largest subnormal number
    short al_val = 0x03ff; 
    short bh_val = 0x03fe;
    short bl_val = 0x03fe;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    asm( "{set.eq.ftz.f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
    *d = val;
}

__global__ void test_set_f16x2_boolop(half2* d) {
    half2 val; 
    short ah_val = 0x3c00;
    short al_val = 0x3c00; 
    short bh_val = 0x3c00;
    short bl_val = 0x3c00;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    asm(".reg .pred %p;\n\t" 
        "setp.eq.s32 %p, 1, 34;\n\t"
        "set.eq.and.f16x2.f16x2 %0,%1,%2,%p;\n\t" 
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); 
    *d = val;
}

__global__ void test_set_f16x2_boolop_ftz(half2* d) {
    half2 val; 
    short ah_val = 0x03ff; // the largest subnormal number
    short al_val = 0x03ff;
    short bh_val = 0x03fe;
    short bl_val = 0x03fe;
    half ah = *reinterpret_cast<half*>(&ah_val);
    half al = *reinterpret_cast<half*>(&al_val);
    half bh = *reinterpret_cast<half*>(&bh_val);
    half bl = *reinterpret_cast<half*>(&bl_val);
    half2 a(ah,al);
    half2 b(bh,bl);
    asm(".reg .pred %p;\n\t" 
        "setp.eq.s32 %p, 1, 34;\n\t"
        "set.eq.xor.ftz.f16x2.f16x2 %0,%1,%2,%p;\n\t" 
        :"=r"(__HALF2_TO_UI(val)) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); 
    *d = val;
}