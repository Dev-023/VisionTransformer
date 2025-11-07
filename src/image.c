#include "../include/image.h"
#include "../include/ViT.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

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
    const float scale = ((float)(IMAGE_SIZE-1)/(IMAGE_SCALING-1));

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

    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &maxval);
    fgetc(fp);

    // printf("Width = %d, Height = %d, Max = %d\n", width, height, maxval);

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

    Tensor4 output = alloc_tensor4(DATASET_BATCH_SIZE, 3, height, width);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // printf("%lf ", ((float)data[(y*width + x)*3 + c])/255.0f);
                T4(output, 0, c, y, x) = ((float)data[(y*width + x)*3 + c])/255.0f;
            }
        }
    }
    free(data);
    return output;
}

Tensor4 Resize256(Tensor4 input) {
    int src_w = input.Y;
    int src_h = input.X;
    int dst_w, dst_h;

    if (src_w>=src_h) {
        dst_h = PREPROCESS_SCALE;
        dst_w = (int)roundf(((float)dst_h/src_h)*src_w);
    }
    else {
        dst_w = PREPROCESS_SCALE;
        dst_h = (int)roundf(((float)dst_w/src_w)*src_h);
    }
    Tensor4 out = alloc_tensor4(DATASET_BATCH_SIZE, 3, dst_h, dst_w); // B=1, C=3, H=dst_h, W=dst_w

    double x_ratio = (double)(src_w) / dst_w;
    double y_ratio = (double)(src_h) / dst_h;

    for (int y = 0; y < dst_h; y++) {

        double sy = (y + 0.5) * y_ratio - 0.5;
        // double sy = (float)y * (src_h - 1) / (dst_h - 1);

        if (sy < 0) sy = 0;
        int y0 = (int)sy;
        int y1 = y0 + 1;
        double wy = sy - y0;
        if (y1 >= src_h) { y1 = src_h - 1; wy = 0.0; }

        for (int x = 0; x < dst_w; x++) {

            double sx = (x + 0.5) * x_ratio - 0.5;
            // double sx = (float)x * (src_w - 1) / (dst_w - 1);

            if (sx < 0) sx = 0;
            int x0 = (int)sx;
            int x1 = x0 + 1;
            double wx = sx - x0;
            if (x1 >= src_w) { x1 = src_w - 1; wx = 0.0; }

            float w00 = (1.0 - wx) * (1.0 - wy);
            float w10 = wx * (1.0 - wy);
            float w01 = (1.0 - wx) * wy;
            float w11 = wx * wy;

            // size_t idx00 = (y0 * src_w + x0) * 3;
            // size_t idx10 = (y0 * src_w + x1) * 3;
            // size_t idx01 = (y1 * src_w + x0) * 3;
            // size_t idx11 = (y1 * src_w + x1) * 3;

            for (int c = 0; c < 3; c++) {
                double v00 = T4(input, 0, c, y0, x0);
                double v10 = T4(input, 0, c, y0, x1);
                double v01 = T4(input, 0, c, y1, x0);
                double v11 = T4(input, 0, c, y1, x1);

                double val = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
                T4(out, 0, c, y, x) = (float)val;
            }
        }
    }

    return out;
}

Tensor4 Crop224(Tensor4 input) {
    Tensor4 out = alloc_tensor4(DATASET_BATCH_SIZE, 3, IMAGE_SCALING, IMAGE_SCALING);
    int top = roundf((input.X-IMAGE_SCALING)/2.0f);
    int left = roundf((input.Y-IMAGE_SCALING)/2.0f);
    // int bottom = input.X - IMAGE_SCALING - top;
    // int right = input.Y - IMAGE_SCALING - left;
    for (int x = 0; x < IMAGE_SCALING; x++) {
        for (int y = 0; y < IMAGE_SCALING; y++) {
            for (int c = 0; c < 3; c++) {
                T4(out, 0, c, y, x) = T4(input, 0, c, top+y, left+x);
            }
        }
    }
    return out;
}

void Normalize(Tensor4 input) {
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};

    for (int x = 0; x < IMAGE_SCALING; x++) {
        for (int y = 0; y < IMAGE_SCALING; y++) {
            for (int c = 0; c < 3; c++) {
                T4(input, 0, c, y, x) = (T4(input, 0, c, y, x) - mean[c])/std[c];
            }
        }
    }
}

Tensor3 Conv2D(Tensor4 Images, Tensor4 Kernel, Tensor1 bias) {
    Tensor3 out = alloc_tensor3(Images.B, IMAGE_SCALING*IMAGE_SCALING/(Kernel.X*Kernel.Y), Kernel.B);

    for (int c = 0; c < Kernel.B; c++) {
        int indx = 0;
        // H = vertical, W = horizontal
        for (int y0 = 0; y0 < IMAGE_SCALING; y0 += Kernel.X) {      // move vertically
            for (int x0 = 0; x0 < IMAGE_SCALING; x0 += Kernel.Y) {  // move horizontally
                float sum = 0.0f;
                int count=0;
                for (int cc = 0; cc < 3; cc++) {
                    for (int ky = 0; ky < Kernel.Y; ky++) {
                        for (int kx = 0; kx < Kernel.X; kx++) {
                            sum += T4(Images, 0, cc, y0 + ky, x0 + kx) * T4(Kernel, c, cc, ky, kx);
                            count++;
                            // printf("%d: %6.4f\n", count, T4(Images, 0, cc, y0 + ky, x0 + kx) * T4(Kernel, c, cc, ky, kx));
                        }
                    }
                }
                T3(out, 0, indx, c) = sum + bias.data[c];
                // printf("\n%10.4f\n\n", sum + bias.data[c]);

                // int tmp;
                // scanf("%d", &tmp);
                
                indx++;
            }
        }
    }
    return out;
}