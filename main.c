#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "include/ViT.h"

void print_tensor4(Tensor4 t) {
    int B = t.B;
    int C = t.H;
    int H = t.Y;
    int W = t.X;
    printf("tensor([[\n");
    for (int b = 0; b < B; b++) {
        printf(" [\n");
        for (int c = 0; c < C; c++) {
            printf("  [\n");
            for (int y = 0; y < H; y++) {
                printf("   [");
                for (int x = 0; x < W; x++) {
                    printf("%.4f", T4(t, b, c, y, x));
                    if (x < W - 1) printf(", ");
                }
                printf("]");
                if (y < H - 1) printf(",\n");
                else printf("\n");
            }
            printf("  ]");
            if (c < C - 1) printf(",\n\n");
            else printf("\n");
        }
        printf(" ]");
        if (b < B - 1) printf(",\n\n");
        else printf("\n");
    }
    printf("]])\n");
}


int main() {
    char *Labels = malloc(sizeof(char) * DATASET_SIZE);
    // (B, C, H, W);
    // Tensor4 Images = LoadCIFAR10Dataset("dataset/cifar-10-batches-bin/train_all.bin", Labels, 0);
    Tensor4 Images = LoadImageFromPPM("dataset/ImageNetSelected/n01687978_10071.ppm");
    print_tensor4(Images);
    Tensor4 ResizedImages = Resize256(Images);
    // print_tensor4(ResizedImages);
    free_tensor4(Images);
    free_tensor4(ResizedImages);
    
}