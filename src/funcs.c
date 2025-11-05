#include "../include/funcs.h"
#include <math.h>
#include <stdio.h>

Tensor4 split_heads(Tensor3 in, int H) {
    int B = in.B, X = in.X, D = in.D;
    int Y = D / H;

    Tensor4 out = alloc_tensor4(B, H, X, Y);

    for(int b=0; b<B; b++)
        for(int x=0; x<X; x++)
            for(int h=0; h<H; h++)
                for(int y=0; y<Y; y++)
                    T4(out,b,h,x,y) = T3(in,b,x,h*Y + y);
    return out;
}

Tensor3 merge_heads(Tensor4 in) {
    int B = in.B, H = in.H, X = in.X, Y = in.Y;
    Tensor3 out = alloc_tensor3(B, X, H*Y);

    for(int b=0; b<B; b++)
        for(int x=0; x<X; x++)
            for(int h=0; h<H; h++)
                for(int y=0; y<Y; y++)
                    T3(out,b,x,h*Y + y) = T4(in,b,h,x,y);

    return out;
}

Tensor4 qk_dot(Tensor4 Q, Tensor4 K) {
    int B=Q.B, H=Q.H, X=Q.X, Y=Q.Y;
    Tensor4 Scores = alloc_tensor4(B,H,X,X);

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++)
                for(int j=0; j<X; j++){
                    float sum = 0.0f;
                    for(int d=0;d<Y;d++)
                        sum += T4(Q,b,h,i,d) * T4(K,b,h,j,d);
                    T4(Scores,b,h,i,j) = sum;
                }
    return Scores;
}

void scale_scores(Tensor4 Scores) {
    float s = 1.0f / sqrtf((float)Scores.Y);
    int B=Scores.B, H=Scores.H, X=Scores.X;

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++)
                for(int j=0; j<X; j++)
                    T4(Scores,b,h,i,j) *= s;
}

void softmax_scores(Tensor4 Scores) {
    int B=Scores.B, H=Scores.H, X=Scores.X;

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++){

                float maxv = -1e30f;
                for(int j=0; j<X; j++)
                    if(T4(Scores,b,h,i,j) > maxv) maxv = T4(Scores,b,h,i,j);

                float sum = 0.0f;
                for(int j=0; j<X; j++){
                    T4(Scores,b,h,i,j) = expf(T4(Scores,b,h,i,j) - maxv);
                    sum += T4(Scores,b,h,i,j);
                }

                for(int j=0; j<X; j++)
                    T4(Scores,b,h,i,j) /= sum;
            }
}


Tensor3 layernorm(Tensor3 in, Tensor1 gamma, Tensor1 beta) {
    int B = in.B, X = in.X, D = in.D;
    Tensor3 Out = alloc_tensor3(B, X, D);
    for (int b = 0; b < B; b++) {
        for (int x = 0; x < X; x++) {
            float mean = 0.0f;
            for (int d = 0; d < D; d++)
                mean += T3(in, b, x, d);
            mean /= (float)D;

            float var = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = T3(in, b, x, d) - mean;
                var += diff * diff;
            }
            var /= (float)D;

            float inv_std = 1.0f / sqrtf(var + EPS);

            for (int d = 0; d < D; d++) {
                float normed = (T3(in, b, x, d) - mean) * inv_std;
                T3(Out, b, x, d) = normed * T1(gamma, d) + T1(beta, d);
            }
        }
    }
    return Out;
}



Tensor4 av_dot(Tensor4 Scores, Tensor4 V) {
    int B=Scores.B, H=Scores.H, X=Scores.X, Y=V.Y;
    Tensor4 Out = alloc_tensor4(B,H,X,Y);

    for(int b=0; b<B; b++)
        for(int h=0; h<H; h++)
            for(int i=0; i<X; i++)
                for(int y=0; y<Y; y++){
                    float sum = 0.0f;
                    for(int j=0; j<X; j++)
                        sum += T4(Scores,b,h,i,j) * T4(V,b,h,j,y);
                    T4(Out,b,h,i,y) = sum;
                }
    return Out;
}

// Compute row-wise stable softmax on Matrix input, store in output
void softmax_matrix(Matrix input, Matrix output) {
    if (input.rows != output.rows || input.cols != output.cols) {
        fprintf(stderr, "Error: softmax_matrix() dimension mismatch\n");
        return;
    }

    for (int r = 0; r < input.rows; r++) {
        // Step 1: find max in this row
        float max_val = -INFINITY;
        for (int c = 0; c < input.cols; c++) {
            if (M(input, r, c) > max_val)
                max_val = M(input, r, c);
        }

        // Step 2: compute exponentials of (x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < input.cols; c++) {
            float e = expf(M(input, r, c) - max_val);
            M(output, r, c) = e;
            sum_exp += e;
        }

        // Step 3: normalize by sum of exps
        for (int c = 0; c < input.cols; c++) {
            M(output, r, c) /= sum_exp;
        }
    }
}

Matrix transpose(Matrix input) {
    Matrix output = alloc_matrix(input.cols, input.rows);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            M(output, j, i) = M(input, i, j);
        }
    }

    free_matrix(input);
    return output;
}

/*
Implementing Matrix multiplication with blocked computing
The input is segmented into blocks of predefined size (size defined by macro at the beginning
Then the computation is processed on the blocks and accumulated in order to get the correct
corresponding matrix product
*/
Matrix multMatrix(Matrix A, Matrix B) {
    if (A.cols != B.rows) {
        printf("Error: matmul_blocked() dimension mismatch: %d×%d * %d×%d\n",
                A.rows, A.cols, B.rows, B.cols);
        exit(1);
    }

    Matrix C = alloc_matrix(A.rows, B.cols);

    int n = A.rows;
    int m = B.cols;
    int p = A.cols;

    // zero-ing the target loc for accumulation
    for (int i = 0; i < n * m; i++) C.data[i] = 0.0f;

    for (int ii = 0; ii < n; ii += BLOCK) {
        for (int kk = 0; kk < p; kk += BLOCK) {
            for (int jj = 0; jj < m; jj += BLOCK) {

                int i_max = (ii + BLOCK < n) ? (ii + BLOCK) : n;
                int k_max = (kk + BLOCK < p) ? (kk + BLOCK) : p;
                int j_max = (jj + BLOCK < m) ? (jj + BLOCK) : m;

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        float a_val = M(A, i, k);
                        float *c_row = &M(C, i, jj);
                        float *b_row = &M(B, k, jj);
                        for (int j = jj; j < j_max; j++) {
                            c_row[j - jj] += a_val * b_row[j - jj];
                        }
                    }
                }
            }
        }
    }

    return C;
}

/*
column-wise addition of matrices specifically programmed of bias addition in MHA
USE WITH CAUTION cuz implementation is oddly specific
*/
Matrix addMatrix_inplace(Matrix A, Matrix B) {
    for (int j = 0; j < A.cols; j++) {
        for (int i = 0; i < A.rows; i++) {
            M(A, i, j) = M(A, i, j) + M(B, j, 0);
        }
    }

    return A;
}

// Getting pointer to specific matrix of the input tensor
Matrix getMatrix(Tensor3 t, int b) {
    Matrix m;
    m.rows = t.X;
    m.cols = t.D;
    m.data = &t.data[b * t.X * t.D];  // direct pointer into Tensor3 memory
    return m;
}

// Getting pointer to specific offset matrix segment of concatenated tensor (Implemented for MHA)
Matrix getOffsetMatrix(Tensor3 t, int b, int colOffset, int cols) {
    Matrix m;
    m.rows = t.X;
    m.cols = cols;
    m.data = &t.data[b * t.X * t.D + colOffset];
    return m;
}

// Computing qkv concatenated heads
Tensor3 MHA(Tensor3 input, Matrix qkvWeight, Matrix qkvBias) {
    qkvWeight = transpose(qkvWeight);
    int B = input.B;
    Tensor3 output = alloc_tensor3(B, input.X, qkvWeight.cols);

    for (int b = 0; b < B; b++) {
        Matrix temp = multMatrix(getMatrix(input, b), qkvWeight);
        addMatrix_inplace(temp, qkvBias);

        Matrix outSlot = getMatrix(output, b);
        memcpy(outSlot.data, temp.data, temp.rows * temp.cols * sizeof(float));
        free_matrix(temp);
    }

    for (int b = 0; b < B; b++) {
        Matrix Q = getOffsetMatrix(output, b, 0, 192);
        Matrix K = getOffsetMatrix(output, b, 192, 192);
        Matrix V = getOffsetMatrix(output, b, 384, 192);

        // Computing Attention
        Matrix Kt = transpose(K);
        Matrix QKt = multMatrix(Q, Kt);

        free_matrix(Kt);
    }
}
