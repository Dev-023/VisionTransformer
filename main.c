#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "include/ViT.h"

int main() {
    char** Labels = load_labels("dataset/ImageNetSelected/Labels.txt", 1000);
    // (B, C, H, W);
    Tensor4 Images = LoadImageFromPPM("dataset/ImageNetSelected/n01687978_10071.ppm");
    // n01735189_11267
    if (PRINT) {
        printf("Input Image\n");
        print_tensor4(Images);
    }

    Tensor4 ResizedImages = Resize256(Images);
    if (PRINT) {
        printf("\n\nResized Image\n");
        print_tensor4(ResizedImages);
    }

    Tensor4 CroppedImages = Crop224(ResizedImages);
    if (PRINT) {
        printf("\n\nCropped Image\n");
        print_tensor4(CroppedImages);
    }

    Normalize(CroppedImages);
    if (PRINT) {
        printf("\n\nNormalized Image\n");
        print_tensor4(CroppedImages);
    }

    Tensor4 patch_embed_proj_weights = GetData4("parameters/patch_embed_proj_weight.bin");
    Tensor1 patch_embed_proj_biases = GetData1("parameters/patch_embed_proj_bias.bin");

    Tensor3 conv = Conv2D(CroppedImages, patch_embed_proj_weights, patch_embed_proj_biases);
    if (PRINT) {
        printf("\n\nConvoltioned Image\n");
        print_tensor3(conv);
    }

    Tensor3 cls = GetData3("parameters/cls_token.bin");
    Tensor3 pos_embed = GetData3("parameters/pos_embed.bin");
    Tensor3 PreprocessedInputs = addCLSToken(conv, cls, pos_embed);
    if (PRINT) {
        printf("\n\nPatch Embedded Image\n");
        print_tensor3(PreprocessedInputs);
    }

    // ========================= PUT INTO LOOPS LATER =========================
    for (int e = 0; e < ENCODER_BLOCKS; e++) {
        if (PRINT) 
            printf("\n\n\n================================================ BLOCK %2d ================================================", e);
        Tensor1 norm_1_weight = GetData1(GetPath(e, "norm1_weight"));
        Tensor1 norm_1_bias = GetData1(GetPath(e, "norm1_bias"));
        Tensor3 ResidualAdd1 = copytensor3(PreprocessedInputs);
        PreprocessedInputs = layernorm(PreprocessedInputs, norm_1_weight, norm_1_bias);
        if (PRINT) {
            printf("\nNorm1ed Image\n");
            print_tensor3(PreprocessedInputs);
        }

        Matrix qkv_weight = GetData2(GetPath(e, "attn_qkv_weight"));
        Tensor1 qkv_bias = GetData1(GetPath(e, "attn_qkv_bias"));
        Matrix proj_weight = GetData2(GetPath(e, "attn_proj_weight"));
        Tensor1 proj_bias = GetData1(GetPath(e, "attn_proj_bias"));
        Tensor3 MHAOutput = MHA(PreprocessedInputs, qkv_weight, qkv_bias, proj_weight, proj_bias);
        if (PRINT) {
            printf("\n\nMHA Output\n");
            print_tensor3(MHAOutput);
        }

        Tensor1 norm_2_weight = GetData1(GetPath(e, "norm2_weight"));
        Tensor1 norm_2_bias = GetData1(GetPath(e, "norm2_bias"));
        MHAOutput = addTensor3(MHAOutput, ResidualAdd1);
        Tensor3 ResidualAdd2 = copytensor3(MHAOutput);
        MHAOutput = layernorm(MHAOutput, norm_2_weight, norm_2_bias);
        if (PRINT) {
            printf("\n\nNorm2ed Image\n");
            print_tensor3(MHAOutput);
        }

        Matrix fc1_weight = GetData2(GetPath(e, "mlp_fc1_weight"));
        Tensor1 fc1_bias = GetData1(GetPath(e, "mlp_fc1_bias"));
        Tensor3 FC1out = mlp_forward(MHAOutput, fc1_weight, fc1_bias);
        if (PRINT) {
            printf("\n\nFC1\n");
            print_tensor3(FC1out);
        }
        
        gelu(FC1out);
        if (PRINT) {
            printf("\n\nGeLU\n");
            print_tensor3(FC1out);
        }

        Matrix fc2_weight = GetData2(GetPath(e, "mlp_fc2_weight"));
        Tensor1 fc2_bias = GetData1(GetPath(e, "mlp_fc2_bias"));
        Tensor3 FC2out = mlp_forward(FC1out, fc2_weight, fc2_bias);
        if (PRINT) {
            printf("\n\nFC2\n");
            print_tensor3(FC2out);
        }

        PreprocessedInputs = addTensor3(FC2out, ResidualAdd2);
        if (PRINT) {
            printf("\n\nBlock %d Output\n", e);
            print_tensor3(PreprocessedInputs);
        }

        free_tensor1(norm_1_weight);
        free_tensor1(norm_1_bias);
        free_tensor1(norm_2_weight);
        free_tensor1(norm_2_bias);
        free_matrix(qkv_weight);
        free_tensor1(qkv_bias);
        free_matrix(proj_weight);
        free_tensor1(proj_bias);
        free_matrix(fc1_weight);
        free_tensor1(fc1_bias);
        free_matrix(fc2_weight);
        free_tensor1(fc2_bias);
        free_tensor3(MHAOutput);
        free_tensor3(ResidualAdd1);
        free_tensor3(FC1out);
        free_tensor3(FC2out);
        free_tensor3(ResidualAdd2);
    }
    // ========================= PUT INTO LOOPS LATER =========================
    Tensor3 cls_token = GetCLSToken(PreprocessedInputs);
    Tensor1 norm_weight = GetData1("parameters/norm_weight.bin");
    Tensor1 norm_bias = GetData1("parameters/norm_bias.bin");
    Tensor3 NormedOutput = layernorm(cls_token, norm_weight, norm_bias);
    if (PRINT) {
        printf("\nNormed Image\n");
        print_tensor3(NormedOutput);
    }

    Matrix head_weight = GetData2("parameters/head_weight.bin");
    Tensor1 head_bias = GetData1("parameters/head_bias.bin");
    Tensor3 Output = mlp_forward(NormedOutput, head_weight, head_bias);
    if (PRINT) {
        printf("\n\nOutput\n");
        print_tensor3(Output);
    }

    for (int b = 0; b < Output.B; b++) {
        int indx = 0;
        float max = T3(Output, b, 0, 0);
        for (int i = 1; i < 1000; i++) {
            if (T3(Output, b, 0, i) > max) {
                indx = i;
                max = T3(Output, b, 0, i);
            }
        }
        printf("Prediction [%d]: %s\n", b, Labels[indx]);
    }
    
    free_tensor1(patch_embed_proj_biases);
    free_tensor3(cls);
    free_tensor3(pos_embed);
    free_tensor3(conv);
    // free_tensor3(PreprocessedInputs);
    free_tensor3(cls_token);
    free_tensor3(NormedOutput);
    free_tensor4(Images);
    free_tensor4(ResizedImages);
    free_tensor4(patch_embed_proj_weights);
    free_tensor4(CroppedImages);
    free_matrix(head_weight);
    free_tensor1(head_bias);
    free_tensor3(Output);
    free(Labels);
}