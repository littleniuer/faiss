#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include "safe_memory.h"
#include <stdio.h>

static inline int init_handle_param(KRLDistanceHandle **kdh, size_t quant_bits, size_t blocksize, size_t codes_num,
    size_t dim, size_t num_base, int metric_type)
{
    (*kdh) = (KRLDistanceHandle *)malloc(sizeof(KRLDistanceHandle));
    if ((*kdh) == NULL) {
        printf("Error: FAILALLOC in init_handle_param\n");
        return FAILALLOC;
    }
    (*kdh)->data_bits = quant_bits;
    (*kdh)->blocksize = blocksize;
    (*kdh)->ny = codes_num;
    (*kdh)->d = dim;
    (*kdh)->M = num_base;
    (*kdh)->metric_type = metric_type;
    return SUCCESS;
}

int krl_create_distance_handle(KRLDistanceHandle **kdh, size_t accu_level, size_t blocksize, size_t codes_num,
    size_t dim, size_t num_base, int metric_type, const uint8_t *codes, size_t codes_size)
{
    if (kdh == NULL || (*kdh) != NULL) {
        printf("Error: INVALPOINTER in krl_create_distance_handle\n");
        return INVALPOINTER;
    }
    if (accu_level < 1 || accu_level > 3 || num_base < 1 || num_base > 65535 ||
        (blocksize != 16 && blocksize != 32 && blocksize != 64) || codes_num < 1 || codes_num > 1ULL << 30 || dim < 1 ||
        dim > 65535 || (metric_type != 0 && metric_type != 1) || codes == NULL ||
        codes_size < num_base * codes_num * dim * 4) {
        printf("Error: INVALPARAM in krl_create_distance_handle\n");
        return INVALPARAM;
    }
    const size_t quant_bits = 4 << accu_level;
    /* Initializes the handle and its parameters. */
    int singal = init_handle_param(kdh, quant_bits, blocksize, codes_num, dim, num_base, metric_type);
    if (singal != SUCCESS) {
        return singal;
    }
    /* The value of ny must be a multiple of the blocksize. */
    (*kdh)->ceil_ny = (codes_num + blocksize - 1) & (-blocksize);
    /* Full-precision matrix-vector multiplication, transpose required */
    if (quant_bits == 32) {
        (*kdh)->quanted_bytes = 0;
        (*kdh)->quanted_codes = NULL;
        (*kdh)->transposed_bytes = num_base * (*kdh)->ceil_ny * dim * sizeof(float);
        (*kdh)->transposed_codes =
            (float *)aligned_alloc(KRL_DEFAULT_ALIGNED, num_base * (*kdh)->ceil_ny * dim * sizeof(float));
        if ((*kdh)->transposed_codes == NULL) {
            krl_clean_distance_handle(kdh);
            printf("Error: FAILALLOC in krl_create_distance_handle\n");
            return FAILALLOC;
        }
        for (size_t m = 0; m < num_base; m++) {
            int ret = krl_matrix_block_transpose((const uint32_t *)(codes + sizeof(float) * m * codes_num * dim),
                codes_num,
                dim,
                blocksize,
                (uint32_t *)((*kdh)->transposed_codes + m * (*kdh)->ceil_ny * dim),
                (*kdh)->ceil_ny * dim * sizeof(uint32_t));
            if (ret != 0) {
                krl_clean_distance_handle(kdh);
                printf("Error: UNSAFEMEM in krl_create_distance_handle\n");
                return UNSAFEMEM;
            }
        }
        /* Quantization matrix-vector multiplication, which needs to be quantized and does not need to be transposed. */
    } else {
        const size_t codes_length = num_base * codes_num * dim;
        (*kdh)->transposed_bytes = 0;
        (*kdh)->transposed_codes = NULL;
        (*kdh)->quanted_bytes = codes_length * (quant_bits >> 3);
        (*kdh)->quanted_codes = (uint8_t *)aligned_alloc(KRL_DEFAULT_ALIGNED, codes_length * (quant_bits >> 3));
        if ((*kdh)->quanted_codes == NULL) {
            krl_clean_distance_handle(kdh);
            printf("Error: FAILALLOC in krl_create_distance_handle\n");
            return FAILALLOC;
        }
        if (quant_bits == 8) {
            (*kdh)->blocksize = 0;
            (*kdh)->quanted_scale = 1;
            (*kdh)->quanted_bias = 0;
            quant_sq8((idx_t)codes_length,
                (const float *)codes,
                (uint8_t *)(*kdh)->quanted_codes,
                metric_type,
                (*kdh)->blocksize,
                (*kdh)->quanted_scale,
                (*kdh)->quanted_bias);
        } else if (quant_bits == 16) {
            (*kdh)->blocksize = 0;
            (*kdh)->quanted_scale = 1;
            (*kdh)->quanted_bias = 0;
            quant_f16((const float *)codes, (idx_t)codes_length, (float16_t *)(*kdh)->quanted_codes);
        }
    }
    return SUCCESS;
}

void krl_clean_distance_handle(KRLDistanceHandle **kdh)
{
    if (kdh == NULL) {
        return;
    }
    if (*kdh) {
        if ((*kdh)->transposed_codes && (*kdh)->transposed_bytes > 0) {
            free((*kdh)->transposed_codes);
            (*kdh)->transposed_codes = NULL;
        }
        if ((*kdh)->quanted_codes && (*kdh)->quanted_bytes > 0) {
            free((*kdh)->quanted_codes);
            (*kdh)->quanted_codes = NULL;
        }
        free(*kdh);
        (*kdh) = NULL;
    }
}

int krl_create_LUT8b_handle(KRLLUT8bHandle **klh, int use_idx, size_t capacity)
{
    if (klh == NULL || (*klh) != NULL) {
        printf("Error: INVALPOINTER in krl_create_LUT8b_handle\n");
        return INVALPOINTER;
    }
    if (use_idx < 0 || use_idx > 1 || capacity < 1) {
        printf("Error: INVALPARAM in krl_create_LUT8b_handle\n");
        return INVALPARAM;
    }
    (*klh) = (KRLLUT8bHandle *)malloc(sizeof(KRLLUT8bHandle));
    if ((*klh) == NULL) {
        printf("Error: FAILALLOC in krl_create_LUT8b_handle\n");
        return FAILALLOC;
    }
    (*klh)->use_idx = use_idx;
    (*klh)->capacity = (capacity + 15) & (-16);
    if (use_idx == 1) {
        (*klh)->idx_buffer = (size_t *)aligned_alloc(KRL_DEFAULT_ALIGNED, capacity * sizeof(size_t));
        if ((*klh)->idx_buffer == NULL) {
            krl_clean_LUT8b_handle(klh);
            printf("Error: FAILALLOC in krl_create_LUT8b_handle\n");
            return FAILALLOC;
        }
    } else {
        (*klh)->idx_buffer = NULL;
    }
    (*klh)->distance_buffer = (float *)aligned_alloc(KRL_DEFAULT_ALIGNED, capacity * sizeof(float));
    if ((*klh)->distance_buffer == NULL) {
        krl_clean_LUT8b_handle(klh);
        printf("Error: FAILALLOC in krl_create_LUT8b_handle\n");
        return FAILALLOC;
    }
    return SUCCESS;
}

void krl_clean_LUT8b_handle(KRLLUT8bHandle **klh)
{
    if (klh == NULL) {
        return;
    }
    if ((*klh)->capacity > 0) {
        if ((*klh)->use_idx > 0 && (*klh)->idx_buffer) {
            free((*klh)->idx_buffer);
            (*klh)->idx_buffer = NULL;
        }
        if ((*klh)->distance_buffer) {
            free((*klh)->distance_buffer);
            (*klh)->distance_buffer = NULL;
        }
        free(*klh);
        (*klh) = NULL;
    }
}

size_t *krl_get_idx_pointer(const KRLLUT8bHandle *klh)
{
    if (klh == NULL) {
        printf("Error: INVALPOINTER in krl_get_idx_pointer\n");
        return NULL;
    } else {
        return klh->idx_buffer;
    }
}

float *krl_get_dist_pointer(const KRLLUT8bHandle *klh)
{
    if (klh == NULL) {
        printf("Error: INVALPOINTER in krl_get_dist_pointer\n");
        return NULL;
    } else {
        return klh->distance_buffer;
    }
}
