#include <limits>

__global__ void test_rsqrt_approx_ftz_f64(double* d, double* a){
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(*a));
}

__global__ void test_rsqrt_approx_ftz_f64_subnormal(double* d){
    long x_bit = 0x000FFFFFFFFFFFFF;
    double x = *reinterpret_cast<double*>(&x_bit);
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(x));
}

__global__ void test_rsqrt_approx_ftz_f64_neg_subnormal(double* d){
    long x_bit = 0x800FFFFFFFFFFFFF;
    double x = *reinterpret_cast<double*>(&x_bit);
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(x));
}

__global__ void test_rsqrt_approx_ftz_f64_inf(double* d){
    long x_bit = 0x7FF0000000000000;
    double x = *reinterpret_cast<double*>(&x_bit);
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(x));
}

__global__ void test_rsqrt_approx_ftz_f64_neg_inf(double* d){
    long x_bit = 0xFFF0000000000000;
    double x = *reinterpret_cast<double*>(&x_bit);
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(x));
}

__global__ void test_rsqrt_approx_ftz_f64_zero(double* d){
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(0.0));
}

__global__ void test_rsqrt_approx_ftz_f64_neg_zero(double* d){
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(-0.0));
}

__global__ void test_rsqrt_approx_ftz_f64_nan(double* d){
    long x_bit = 0x7FF8000000000000;
    double x = *reinterpret_cast<double*>(&x_bit);
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(*d) : "d"(x));
}
