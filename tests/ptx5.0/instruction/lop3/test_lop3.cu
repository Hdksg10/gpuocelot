// __device__ int my_LOP_0x54(int A, int B, int C){
//   int temp;
//   asm("lop3.b32 %0, %1, %2, %3, 0x54;" : "=r"(temp) : "r"(A), "r"(B), "r"(C));
//   return temp;
// }

// __global__ void testkernel(){

//   printf("A=true, B=false, C=true,   F=%d\n", my_LOP_0x54(true, false, true));
//   printf("A=true, B=false, C=false,  F=%d\n", my_LOP_0x54(true, false, false));
//   printf("A=false, B=false, C=false, F=%d\n", my_LOP_0x54(false, false, false));
// }

__global__ void test_lop3(unsigned* a, unsigned* b, unsigned* c, unsigned* d){
    asm("lop3.b32 %0, %1, %2, %3, 0xff;" : "=r"(*d) : "r"(*a), "r"(*b), "r"(*c));
}