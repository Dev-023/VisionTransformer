#include "../include/image.h"
#include "../include/ViT.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// old funcs

Tensor4 LoadCIFAR10Dataset(const char *Path, char *Labels, int Batch) {
    FILE *f = fopen(Path, "rb");
    if (!f) { printf("File not found\n"); exit(0); }

    fseek(f, Batch*DATASET_BATCH_SIZE*(IMAGE_SIZE*IMAGE_SIZE*3+1), SEEK_CUR);

    Tensor4 Images = alloc_tensor4(DATASET_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE);

    unsigned char l;
    unsigned char buffer[IMAGE_SIZE*IMAGE_SIZE*3];
    for (int b = 0; b < DATASET_BATCH_SIZE; b++) {
        if (fread(&l, 1, 1, f) != 1) {
            printf("End of file or read error\n");
            exit(0);
        }
        Labels[b] = l;

        if (fread(buffer, 1, IMAGE_SIZE*IMAGE_SIZE*3, f) != IMAGE_SIZE*IMAGE_SIZE*3) {
            printf("End of file or read error\n");
            exit(0);
        }
        for (int rgb = 0; rgb < 3; rgb++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                for (int y = 0; y < IMAGE_SIZE; y++) {
                    T4(Images, b, rgb, x, y) = buffer[rgb*IMAGE_SIZE*IMAGE_SIZE + x*IMAGE_SIZE + y]/255.0f;
                }
            }
        }
    }
    return Images;
}

Tensor4 ResizeTo224(Tensor4 input) {

    Tensor4 output = alloc_tensor4(DATASET_BATCH_SIZE, 3, IMAGE_SCALING, IMAGE_SCALING);
    const float scale = (float)((IMAGE_SIZE-1)/(IMAGE_SCALING-1));

    for (int b=0; b<DATASET_BATCH_SIZE; b++)
        for (int c=0; c<3; c++)
            for (int x=0; x<IMAGE_SCALING; x++) {

                float x_in = x * scale;
                int x0 = (int)x_in;
                int x1 = (x0 < IMAGE_SIZE - 1) ? x0 + 1 : x0;
                float dx = x_in - x0;

                for (int y=0; y<IMAGE_SCALING; y++) {
                    float y_in = y * scale;
                    int y0 = (int)y_in;
                    int y1 = (y0 < IMAGE_SIZE - 1) ? y0 + 1 : y0;
                    float dy = y_in - y0;

                    float p00 = T4(input, b, c, x0, y0);
                    float p01 = T4(input, b, c, x0, y1);
                    float p10 = T4(input, b, c, x1, y0);
                    float p11 = T4(input, b, c, x1, y1);

                    float v0 = p00 * (1 - dy) + p01 * dy;
                    float v1 = p10 * (1 - dy) + p11 * dy;
                    float val = v0 * (1 - dx) + v1 * dx;

                    T4(output, b, c, x, y) = val;
                }
            }
    return output;
}

Tensor3 MakePatches(Tensor4 input) {

    int NumPatches = NUM_TOKENS * NUM_TOKENS;
    Tensor3 output = alloc_tensor3(DATASET_BATCH_SIZE, NumPatches, 3*PATCH_SIZE*PATCH_SIZE);

    for (int b = 0; b < DATASET_BATCH_SIZE; b++) {
        int p = 0;
        for (int py = 0; py < IMAGE_SCALING; py += PATCH_SIZE) {
            for (int px = 0; px < IMAGE_SCALING; px += PATCH_SIZE) {

                for (int c = 0; c < 3; c++) {
                    for (int dy = 0; dy < PATCH_SIZE; dy++) {
                        for (int dx = 0; dx < PATCH_SIZE; dx++) {

                            int flat = dy * PATCH_SIZE + dx;
                            T3(output, b, p, c*PATCH_SIZE*PATCH_SIZE + flat) = T4(input, b, c, py + dy, px + dx);
                        }
                    }
                }
                p++;
            }
        }
    }

    return output;
}


// new funcs
Tensor4 LoadImageFromPPM(const char *Path) {
    int width, height, maxval;

    FILE *fp = fopen(Path, "rb");
    if (!fp) {
        perror("Could not open file");
        exit(1);
    }
    char magic[3];
     if (fscanf(fp, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '6') {
        printf("Not a P6 PPM\n");
        exit(1);
    }

    // Skip comments and whitespace before width/height
    int c;
    c = fgetc(fp);
    while (c == '#') {              // comment line
        while ((c = fgetc(fp)) != '\n' && c != EOF);
        c = fgetc(fp);
    }
    ungetc(c, fp);

    // Read width/height
    fscanf(fp, "%d %d", &width, &height);

    // Read maxval
    fscanf(fp, "%d", &maxval);

    // **Consume exactly one whitespace character after maxval**
    fgetc(fp);

    printf("Width = %d, Height = %d, Max = %d\n", width, height, maxval);
    // Allocate pixel buffer
    size_t size = (width) * (height) * 3;
    unsigned char *data = (unsigned char*)malloc(size);
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        exit(1);
    }
    // Read pixel data
    if (fread(data, 1, size, fp) != size) {
        fprintf(stderr, "Unexpected EOF while reading pixel data\n");
        free(data);
        fclose(fp);
        exit(1);
    }
    fclose(fp);

    Tensor4 output = alloc_tensor4(DATASET_BATCH_SIZE, 3, width, height);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                T4(output, 0, c, y, x) = (float) ((int)data[(y*width + x)*3 + c]/255.0f);
            }
        }
    }
    return output;
}

Tensor4 Resize256(Tensor4 input) {
    int row = input.Y, col = input.X;
    if (input.X>input.Y)
        col = PREPROCESS_SCALE;
    else
        row = PREPROCESS_SCALE;

    Tensor4 output = alloc_tensor4(DATASET_BATCH_SIZE, 3, row, col);
    const float scale_x = (float)((IMAGE_SIZE-1)/(col-1));
    const float scale_y = (float)((IMAGE_SIZE-1)/(row-1));

    for (int b=0; b<DATASET_BATCH_SIZE; b++)
        for (int c=0; c<3; c++)
            for (int x=0; x<col; x++) {

                float x_in = x * scale_x;
                int x0 = (int)x_in;
                int x1 = (x0 < IMAGE_SIZE - 1) ? x0 + 1 : x0;
                float dx = x_in - x0;

                for (int y=0; y<row; y++) {
                    float y_in = y * scale_y;
                    int y0 = (int)y_in;
                    int y1 = (y0 < IMAGE_SIZE - 1) ? y0 + 1 : y0;
                    float dy = y_in - y0;

                    float p00 = T4(input, b, c, x0, y0);
                    float p01 = T4(input, b, c, x0, y1);
                    float p10 = T4(input, b, c, x1, y0);
                    float p11 = T4(input, b, c, x1, y1);

                    float v0 = p00 * (1 - dy) + p01 * dy;
                    float v1 = p10 * (1 - dy) + p11 * dy;
                    float val = v0 * (1 - dx) + v1 * dx;

                    T4(output, b, c, x, y) = val/255.0f;
                }
            }
    return output;
}