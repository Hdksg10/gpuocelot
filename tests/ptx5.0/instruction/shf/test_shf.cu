__global__ void test_shf_l_clamp(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}

__global__ void test_shf_r_clamp(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}

__global__ void test_shf_l_wrap(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}

__global__ void test_shf_r_wrap(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}