#include "../include/funcs.h"
#include "../include/ViT.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

void print_tensor4(Tensor4 t) {
    printf("(%d, %d, %d, %d)\n", t.B, t.H, t.X, t.Y);
    int lim = 3;
    int B = t.B;
    int C = t.H;
    int H = t.X;
    int W = t.Y;
    printf("[\n");
    for (int b = 0; b < B; b++) {
        printf("    [\n");
        for (int c = 0; c < C; c++) {
            printf("        [\n");
            for (int y = 0; y < H; y++) {
                if (y>=lim && y<H-lim) {
                    printf("            ...,\n");
                    y=H-lim;
                }
                printf("            [");
                for (int x = 0; x < W; x++) {
                    if (x>=lim && x<W-lim) {
                        printf("..., ");
                        x=W-lim;
                    }
                    printf("%7.4f", T4(t, b, c, y, x));
                    if (x < W - 1) printf(", ");
                }
                printf("]");
                if (y < H - 1) printf(",\n");
                else printf("\n");
            }
            printf("        ]");
            if (c < C - 1) printf(",\n\n");
            else printf("\n");
        }
        printf("    ]");
        if (b < B - 1) printf(",\n\n");
        else printf("\n");
    }
    printf("]\n");
}

void print_tensor3(Tensor3 t) {
    int lim = 3;
    int B = t.B;
    int X = t.X;
    int D = t.D;

    printf("(%d, %d, %d)\n", t.B, t.X, t.D);
    printf("[\n");
    for (int b = 0; b < B; b++) {
        printf("  [\n");
        for (int x = 0; x < X; x++) {

            if (x == lim && X > 2 * lim) {
                printf("    ...,\n");
                x = X - lim;
            }

            printf("    [");
            for (int d = 0; d < D; d++) {

                if (d == lim && D > 2 * lim) {
                    printf("..., ");
                    d = D - lim;
                }

                printf("%7.4f", T3(t, b, x, d));
                if (d < D - 1) printf(", ");
            }
            printf("]");

            if (x < X - 1) printf(",\n");
            else printf("\n");
        }
        printf("  ]");
        if (b < B - 1) printf(",\n\n");
        else printf("\n");
    }
    printf("]\n");
}

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

void scale_scores(Tensor4 Scores, int d_k) {
    float s = 1.0f / sqrtf((float)d_k);
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
            float M2 = 0.0f;
            for (int d = 0; d < D; d++) {
                float val = T3(in, b, x, d);
                float delta = val - mean;
                float count = (float)(d + 1);
                mean += delta / count;
                float delta2 = val - mean;
                M2 += delta * delta2;
            }

            float var = M2 / (float)D;
            float inv_std = 1.0f / sqrtf(var + EPS);

            for (int d = 0; d < D; d++) {
                float val = T3(in, b, x, d);
                float normed = (val - mean) * inv_std;
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

// Compute row-wise stable softmax on Matrix input
void softmax_inplace(Matrix input) {
    for (int r = 0; r < input.rows; r++) {
        // Step 1: Find max value in row for numerical stability
        float max_val = -INFINITY;
        
        for (int c = 0; c < input.cols; c++) {
            float val = M(input, r, c);
            if (val > max_val)
                max_val = val;
        }

        // Step 2: Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int c = 0; c < input.cols; c++) {
            float e = expf(M(input, r, c) - max_val);
            M(input, r, c) = e;  // overwrite with exponent
            sum_exp += e;
        }

        // Step 3: Normalize in-place
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < input.cols; c++) {
            M(input, r, c) *= inv_sum/8;
        }
    }
}

Matrix transpose_matrix(Matrix W) {
    Matrix T = alloc_matrix(W.cols, W.rows);
    for (int r = 0; r < W.rows; r++) 
        for (int c = 0; c < W.cols; c++)
            M(T, c, r) = M(W, r, c);
    return T;
}

Tensor3 gemm(Tensor3 A, Matrix W) {
    int B = A.B;
    int S = A.X;
    int D_in = A.D; 
    int D_out = W.cols;

    Tensor3 Out = alloc_tensor3(B, S, D_out);

    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++)
            for (int d_out = 0; d_out < D_out; d_out++) {
                float sum = 0.0f;
                for (int d_in = 0; d_in < D_in; d_in++) 
                    sum += T3(A, b, s, d_in) * M(W, d_in, d_out);
                T3(Out, b, s, d_out) = sum;
            }
        
    return Out;
}

void add_bias(Tensor3 A, Tensor1 bias) {
    int B = A.B, X = A.X, D = A.D;
    for (int b = 0; b < B; b++) 
        for (int x = 0; x < X; x++)
            for (int d = 0; d < D; d++)
                T3(A, b, x, d) += T1(bias, d);
}

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(Tensor3 A) {
    size_t total_elements = (size_t)A.B * A.X * A.D;

    for (size_t i = 0; i < total_elements; i++) {
        float x = A.data[i];
        float cube = 0.044715f * x * x * x;
        float inner = 0.79788456f * (x + cube);
        A.data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

Tensor3 mlp_forward(Tensor3 in, Matrix weights, Tensor1 biases) {
    Matrix Wt = transpose_matrix(weights);
    
    Tensor3 out = gemm(in, Wt);
    add_bias(out, biases);
    free_matrix(Wt);
    
    return out;
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
Tensor4 addTensor4(Tensor4 A, Tensor4 B) {
    for (int b = 0; b < A.B; b++) {
        for (int h = 0; h < A.H; h++) {
            for (int x = 0; x < A.X; x++) {
                for (int y = 0; y < A.X; y++) {
                    T4(A, b, h, x, y)+=T4(B, b, h, x, y);
                }
            }
        }
    }
    return A;
}

Tensor3 addTensor3(Tensor3 A, Tensor3 B) {
    for (int b = 0; b < A.B; b++) {
        for (int x = 0; x < A.X; x++) {
            for (int d = 0; d < A.D; d++) {
                T3(A, b, x, d)+=T3(B, b, x, d);
            }
        }
    }
    return A;
}

Tensor3 MHA(Tensor3 input, Matrix qkvWeight, Tensor1 qkvBias, Matrix ProjW, Tensor1 ProjB) {
    int B = input.B;
    int X = input.X;
    int D = input.D;
    int D_k = D/NUM_HEAD;

    Matrix QKVWt = transpose_matrix(qkvWeight);
    Tensor3 qkv = gemm(input, QKVWt);
    add_bias(qkv, qkvBias);

    Tensor3 Q = alloc_tensor3(B, X, D);
    Tensor3 K = alloc_tensor3(B, X, D);
    Tensor3 V = alloc_tensor3(B, X, D);

    for (int b = 0; b < B; b++) {
        for (int x = 0; x < X; x++) {
            for (int d = 0; d < D; d++) {
                T3(Q, b, x, d) = T3(qkv, b, x, d);
                T3(K, b, x, d) = T3(qkv, b, x, D + d);
                T3(V, b, x, d) = T3(qkv, b, x, 2*D + d);
            }
        }
    }

    Tensor4 Qh = split_heads(Q, NUM_HEAD);
    Tensor4 Kh = split_heads(K, NUM_HEAD);
    Tensor4 Vh = split_heads(V, NUM_HEAD);

    Tensor4 Scores = qk_dot(Qh, Kh);
    scale_scores(Scores, D_k);
    softmax_scores(Scores);
    Tensor4 Attended = av_dot(Scores, Vh);
    Tensor3 Context = merge_heads(Attended);

    Matrix ProjWt = transpose_matrix(ProjW);
    Tensor3 Out = gemm(Context, ProjWt);
    add_bias(Out, ProjB);

    free_tensor4(Qh);
    free_tensor4(Kh);
    free_tensor4(Vh);
    free_tensor4(Scores);
    free_tensor4(Attended);
    free_tensor3(Context);
    free_matrix(ProjWt);
    free_matrix(QKVWt);


    return Out;
}

Tensor3 GetCLSToken(Tensor3 t) {
    Tensor3 out = alloc_tensor3(t.B, 1, t.D);
    for (int b = 0; b < t.B; b++) {
        for (int i = 0; i < t.D; i++) {
            T3(out, b, 0, i) = T3(t, b, 0, i);
        }
    }
    return out;
}

char** load_labels(const char *path, int num_labels) {
    FILE *fp = fopen(path, "r");
    if (!fp) { printf("Labels file not found!\n"); exit(1); }

    char **labels = malloc(num_labels * sizeof(char*));
    char buffer[256];
    int i = 0;

    while (fgets(buffer, sizeof(buffer), fp) && i < num_labels) {
        buffer[strcspn(buffer, "\n")] = 0; // strip newline
        labels[i] = strdup(buffer);
        i++;
    }

    fclose(fp);
    return labels;
}
