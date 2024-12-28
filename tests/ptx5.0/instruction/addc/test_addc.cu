__global__ void test_addc(unsigned long* d) {
    unsigned long result = 0;  

    asm(
        "addc.cc.u64 %0, %1, %2;\n\t"
        "addc.u64 %0, 0, 0;\n\t"          
        : "=l"(result)                       
        : "l"(5l), "l"(0xffffffffffffffff)                              
    );

    *d = result;
}