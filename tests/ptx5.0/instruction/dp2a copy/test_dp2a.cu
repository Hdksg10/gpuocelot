__global__ void test_dp2a_hi(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("dp2a.hi.u32.u32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}
__global__ void test_dp2a_lo(int32_t* a, unsigned* b, int32_t* c, int32_t* d){
    asm("dp2a.lo.u32.s32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}