#ifndef ViT
#define ViT

#include "structs.h"
#include "image.h"
#include "funcs.h"
#include "parameters.h"

#define PRINT 1
#define IMAGE_SIZE 32               // 32
#define PATCH_SIZE 16               // 4
#define IMAGE_SCALING 224
#define NUM_HEAD 3
#define PREPROCESS_SCALE 256
#define ENCODER_BLOCKS 12           // 12

#define DATASET_BATCH_SIZE 1
#define NUM_BATCHES (DATASET_SIZE/DATASET_BATCH_SIZE)
#define NUM_TOKENS 196
#define TOKEN_SIZE 192
#define HEAD_SIZE 64



#endif