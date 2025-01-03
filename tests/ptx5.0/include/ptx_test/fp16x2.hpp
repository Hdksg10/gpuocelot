#ifndef FP16X2_H
#define FP16X2_H

#include <cuda_fp16.h>

class fp16x2_t {
    public:
        __half x;
        __half y;
    public:
        fp16x2_t() : x(0.0f), y(0.0f) {};
        fp16x2_t(__half x, __half y) : x(x), y(y) {};
};

#endif /* FP16X2_H */
