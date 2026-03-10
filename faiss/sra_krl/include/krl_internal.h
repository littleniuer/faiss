#ifndef KRL_INTERNAL_H
#define KRL_INTERNAL_H

#include "krl.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>
#include <math.h>

typedef int64_t idx_t;

typedef struct KRLBatchDistanceHandle {
    int metric_type;
    float quanted_scale;
    float quanted_bias;
    size_t data_bits;
    size_t full_data_bits;
    size_t M;
    size_t blocksize;
    size_t d;
    size_t ny;
    size_t ceil_ny;
    size_t quanted_bytes;
    size_t transposed_bytes;
    uint8_t *quanted_codes;
    float *transposed_codes;
} KRLDistanceHandle;

typedef struct KRLLookupTable8bitHandle {
    int use_idx;
    size_t capacity;
    size_t *idx_buffer;
    float *distance_buffer;
} KRLLUT8bHandle;

#ifdef __cplusplus
extern "C" {
#endif

int krl_matrix_block_transpose(
    const uint32_t *src, size_t ny, size_t dim, size_t blocksize, uint32_t *block, size_t block_size);

void quant_f16(const float *src, idx_t n, float16_t *out);

void quant_u8(const float *src, idx_t n, uint8_t *out);

void quant_u8_with_parm(const float *src, idx_t n, uint8_t *out, float scale, float bias);

void quant_s8(const float *src, idx_t n, int8_t *out);

void quant_s8_with_parm(const float *src, idx_t n, int8_t *out, float scale);

size_t compute_quant_parm(idx_t n, const float *x, int metric_type, int range, float *scale, float *bias);

void quant_sq8(idx_t n, const float *x, uint8_t *out, int metric_type, int use_parm, float scale, float bias);

#ifdef __cplusplus
}
#endif

#endif  // KRL_INTERNAL_H