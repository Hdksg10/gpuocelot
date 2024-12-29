
__global__ void test_madc_lo_u32(unsigned* d) {
    unsigned result = 0;  

    asm(
        "madc.lo.cc.u32 %0, %1, %2, %3;\n\t" 
        "madc.lo.u32 %0, %1, %2, 0;\n\t"           
        : "=r"(result)                       
        : "r"(2), "r"(2), "r"(0xffffffff)                              
    );

    *d = result;
}

__global__ void test_madc_hi_u32(unsigned* d) {
    unsigned result = 0;  

    asm(
        "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" 
        "madc.hi.u32 %0, %1, %2, 0;\n\t"           
        : "=r"(result)                       
        : "r"(0x20000), "r"(0x20000), "r"(0xffffffff)                              
    );

    *d = result;
}

__global__ void test_madc_lo_u64(unsigned long* d) {
    unsigned long result = 0;  

    asm(
        "madc.lo.cc.u64 %0, %1, %2, %3;\n\t" 
        "madc.lo.u64 %0, %1, %2, 0;\n\t"           
        : "=l"(result)                       
        : "l"(0x2l), "l"(0x2l), "l"(0xffffffffffffffffl)                              
    );

    *d = result;
}

__global__ void test_madc_hi_u64(unsigned long* d) {
    unsigned long result = 0;  

    asm(
        "madc.hi.cc.u64 %0, %1, %2, %3;\n\t" 
        "madc.hi.u64 %0, %1, %2, 0;\n\t"           
        : "=l"(result)                       
        : "l"(0x200000000l), "l"(0x200000000l), "l"(0xffffffffffffffffl)                              
    );

    *d = result;
}