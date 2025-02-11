#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

/* Macros to allow half & half2 to be used by inline assembly.
 * Copy from cuda_fp16.hpp
 */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

__global__ void test_setp_f16(int* d, half* a, half* b) {
    int val = 0; 
    asm(".reg .pred %p;\n\t" 
        "setp.ne.f16 %p,%1,%2;\n\t" 
        "@%p mov.u32 %0, 1;\n\t"
        :"=r"(val) : "h"(__HALF_TO_CUS(*a)),"h"(__HALF_TO_CUS(*b))); \
    *d = val;
}

__global__ void test_setp_f16_ftz(int* d) {
    int val; 
    short a_val = 0x03ff; // the largest subnormal number
    short b_val = 0x03fe;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    asm(".reg .pred %p;\n\t"  
        "setp.eq.ftz.f16 %p,%1,%2;\n\t" 
        "@%p mov.u32 %0, 1;\n\t"
        :"=r"(val) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
    *d = val;
}

__global__ void test_setp_f16_boolop(int* d) {
    int val; 
    short a_val = 0x3c00;
    short b_val = 0x3c00;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    asm(".reg .pred %p;\n\t" 
        ".reg .pred %r;\n\t"
        "setp.eq.s32 %r, 1, 34;\n\t"
        "setp.eq.and.f16 %p,%1,%2,%r;\n\t" 
        "@%p mov.u32 %0, 1;\n\t"
        :"=r"(val) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); 
    *d = val;
}

__global__ void test_setp_f16_boolop_ftz(int* d) {
    int val; 
    short a_val = 0x03ff; // the largest subnormal number
    short b_val = 0x03fe;
    half a = *reinterpret_cast<half*>(&a_val);
    half b = *reinterpret_cast<half*>(&b_val);
    asm(".reg .pred %p;\n\t" 
        ".reg .pred %r;\n\t"
        "setp.eq.s32 %r, 1, 34;\n\t"
        "setp.eq.xor.ftz.f16 %p,%1,%2,%r;\n\t" 
        "@%p mov.u32 %0, 1;\n\t"
        :"=r"(val) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); 
    *d = val;
}

__global__ void test_setp_f16x2(int* d, half2* a, half2* b) {
    int val;  
    asm(".reg .pred %p;\n\t" 
        ".reg .pred %q;\n\t" 
        ".reg .u32 %ra;\n\t"
        ".reg .u32 %rb;\n\t"
        "setp.ne.f16x2 %p|%q,%1,%2;\n\t" 
        "@%p mov.u32 %ra, 1;\n\t"
        "@%q mov.u32 %rb, 2;\n\t"
        "add.u32 %0, %ra, %rb;\n\t"
        :"=r"(val) : "r"(__HALF2_TO_CUI(*a)),"r"(__HALF2_TO_CUI(*b))); \
    *d = val;
}

__global__ void test_setp_f16x2_ftz(int* d) {
    int val; 
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
        ".reg .pred %q;\n\t" 
        ".reg .u32 %ra;\n\t"
        ".reg .u32 %rb;\n\t"
        "setp.eq.ftz.f16x2 %p|%q,%1,%2;\n\t" 
        "@%p mov.u32 %ra, 1;\n\t"
        "@%q mov.u32 %rb, 2;\n\t"
        "add.u32 %0, %ra, %rb;\n\t"
        :"=r"(val) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); \
    *d = val;
}

__global__ void test_setp_f16x2_boolop(int* d) {
    int val; 
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
        ".reg .pred %q;\n\t"
        ".reg .pred %r;\n\t"  
        ".reg .u32 %ra;\n\t"
        ".reg .u32 %rb;\n\t"
        "setp.eq.s32 %r, 1, 34;\n\t"
        "setp.eq.xor.ftz.f16x2 %p|%q,%1,%2,%r;\n\t" 
        "@%p mov.u32 %ra, 1;\n\t"
        "@%q mov.u32 %rb, 2;\n\t"
        "add.u32 %0, %ra, %rb;\n\t"
        :"=r"(val) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); 
    *d = val;
}

__global__ void test_setp_f16x2_boolop_ftz(int* d) {
    int val; 
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
        ".reg .pred %q;\n\t"
        ".reg .pred %r;\n\t"  
        ".reg .u32 %ra;\n\t"
        ".reg .u32 %rb;\n\t"
        "setp.eq.s32 %r, 1, 34;\n\t"
        "setp.eq.xor.ftz.f16x2 %p|%q,%1,%2,%r;\n\t" 
        "@%p mov.u32 %ra, 1;\n\t"
        "@%q mov.u32 %rb, 2;\n\t"
        "add.u32 %0, %ra, %rb;\n\t"
        :"=r"(val) : "r"(__HALF2_TO_CUI(a)),"r"(__HALF2_TO_CUI(b))); 
    *d = val;
}