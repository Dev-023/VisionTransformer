#include "../include/structs.h"

// cpu loads data in blocks
// if not aligned well, its slower cuz otherwise cpu can load it in a single instruction
// generally use 32 bits ka alignment
// avx loads 8 floats at a time in 256 bit registers (float32)
// avx = advanced vector extensions
// avx is cpu version of gpu
inline float* alloc_aligned(size_t n) {
    void *ptr = NULL;
    posix_memalign(&ptr, 32, n * sizeof(float));
    return (float*)ptr;
}

Tensor1 alloc_tensor1(int D) {
    Tensor1 t;
    t.D = D;
    t.data = alloc_aligned((size_t)D);
    return t;
}

Tensor3 alloc_tensor3(int B, int X, int D) {
    Tensor3 t;
    t.B = B;
    t.X = X;
    t.D = D;
    t.data = alloc_aligned((size_t)B*X*D);
    return t;
}

Tensor4 alloc_tensor4(int B, int H, int X, int Y) {
    Tensor4 t;
    t.B = B;
    t.H = H;
    t.X = X;
    t.Y = Y;
    t.data = alloc_aligned((size_t)B*H*X*Y);
    return t;
}

void free_tensor1(Tensor1 t) { free(t.data); }
void free_tensor3(Tensor3 t) {free(t.data);}
void free_tensor4(Tensor4 t) {free(t.data);}