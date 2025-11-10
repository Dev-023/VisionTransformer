#ifndef FUNC_H
#define FUNC_H

#include "structs.h"

#define BLOCK 64

void print_tensor4(Tensor4);
void print_tensor3(Tensor3);

Tensor4 split_heads(Tensor3 in, int H);
Tensor3 merge_heads(Tensor4 in);
Tensor4 qk_dot(Tensor4 Q, Tensor4 K);
void scale_scores(Tensor4 Scores, int d_k); // <-- FIXED
void softmax_scores(Tensor4 Scores);
Tensor4 av_dot(Tensor4 Scores, Tensor4 V);

Tensor3 layernorm(Tensor3, Tensor1, Tensor1);

Matrix transpose_matrix(Matrix W);
Tensor3 gemm(Tensor3 A, Matrix W);
void add_bias(Tensor3 A, Tensor1 bias);
void gelu(Tensor3 A);
Tensor3 mlp_forward(Tensor3 in, Matrix weights, Tensor1 biases);
Tensor3 MHA(Tensor3 input, Matrix qkvWeight, Tensor1 qkvBias, Matrix ProjW, Tensor1 ProjB);
Tensor3 addTensor3(Tensor3, Tensor3);
Tensor4 addTensor4(Tensor4, Tensor4);
Tensor3 GetCLSToken(Tensor3 t);
char** load_labels(const char *path, int num_labels);


#endif