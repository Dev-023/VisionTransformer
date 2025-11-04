#ifndef STRUCT_H
#define STRUCT_H

#include <stdlib.h>

typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    int B, X, D;
    float *data;
} Tensor3;

typedef struct {
    int B, H, X, Y;
    float *data;
} Tensor4;

typedef struct {
    int D;
    float *data;
} Tensor1;


#define EPS 1e-5f
#define M(mat, r, c) mat.data[(r) * mat.cols + (c)]
#define T1(t,d) (t.data[d])
#define T3(t,b,x,d) (t.data[ ((b)*(t.X) + (x))*(t.D) + (d) ])
#define T4(t,b,h,x,y) (t.data[ (((b)*(t.H) + (h))*(t.X) + (x))*(t.Y) + (y) ])

float* alloc_aligned(size_t);
Tensor1 alloc_tensor1(int);
Tensor3 alloc_tensor3(int, int, int);
Tensor4 alloc_tensor4(int, int, int, int);

void free_tensor1(Tensor1);
void free_tensor3(Tensor3);
void free_tensor4(Tensor4);


#endif