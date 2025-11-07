#ifndef FUNC_H
#define FUNC_H

#include "structs.h"

#define BLOCK 64

void print_tensor4(Tensor4);
void print_tensor3(Tensor3);
Tensor3 layernorm(Tensor3, Tensor1, Tensor1);

#endif