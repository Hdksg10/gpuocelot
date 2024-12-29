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