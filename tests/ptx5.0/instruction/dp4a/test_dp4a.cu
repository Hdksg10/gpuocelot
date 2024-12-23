__global__ void test_dp4a_uu(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}
__global__ void test_dp4a_us(int32_t* a, unsigned* b, int32_t* c, int32_t* d){
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}
__global__ void test_dp4a_ss(int* a, int* b, int* c, int* d){
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}