#ifndef KRL_H
#define KRL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define KRL_API_PUBLIC __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KRLBatchDistanceHandle KRLDistanceHandle;

KRL_API_PUBLIC int krl_create_distance_handle(KRLDistanceHandle **kdh, size_t accu_level, size_t blocksize,
    size_t codes_num, size_t dim, size_t num_base, int metric_type, const uint8_t *codes, size_t codes_size);

KRL_API_PUBLIC void krl_clean_distance_handle(KRLDistanceHandle **kdh);

typedef struct KRLLookupTable8bitHandle KRLLUT8bHandle;

KRL_API_PUBLIC int krl_create_LUT8b_handle(KRLLUT8bHandle **klh, int use_idx, size_t capacity);

KRL_API_PUBLIC void krl_clean_LUT8b_handle(KRLLUT8bHandle **klh);

KRL_API_PUBLIC size_t *krl_get_idx_pointer(const KRLLUT8bHandle *klh);

KRL_API_PUBLIC float *krl_get_dist_pointer(const KRLLUT8bHandle *klh);

KRL_API_PUBLIC int krl_L2sqr(const float *x, const float *__restrict y, const size_t d, float *dis, size_t dis_size);

KRL_API_PUBLIC int krl_L2sqr_f16f32(
    const uint16_t *x, const uint16_t *__restrict y, size_t d, float *dis, size_t dis_size);

KRL_API_PUBLIC int krl_L2sqr_u8u32(
    const uint8_t *x, const uint8_t *__restrict y, size_t d, uint32_t *dis, size_t dis_size);

KRL_API_PUBLIC int krl_ipdis(const float *x, const float *__restrict y, const size_t d, float *dis, size_t dis_size);

KRL_API_PUBLIC int krl_L2sqr_ny(float *dis, const float *x, const float *y, size_t ny, size_t d, size_t dis_size);

KRL_API_PUBLIC int krl_L2sqr_ny_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d, size_t dis_size);

KRL_API_PUBLIC int krl_L2sqr_ny_u8f32(
    float *dis, const uint8_t *x, const uint8_t *y, size_t ny, size_t d, size_t dis_size);

KRL_API_PUBLIC int krl_L2sqr_ny_with_handle(
    const KRLDistanceHandle *kdh, float *dis, const float *x, size_t dis_size, size_t x_size);

KRL_API_PUBLIC int krl_inner_product_ny(
    float *dis, const float *x, const float *y, size_t ny, size_t d, size_t dis_size);

KRL_API_PUBLIC int krl_inner_product_ny_f16f32(
    float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d, size_t dis_size);

KRL_API_PUBLIC int krl_inner_product_ny_s8f32(
    float *dis, const int8_t *x, const int8_t *y, size_t ny, size_t d, size_t dis_size);

KRL_API_PUBLIC int krl_inner_product_ny_with_handle(
    const KRLDistanceHandle *kdh, float *dis, const float *x, size_t dis_size, size_t x_size);

KRL_API_PUBLIC int krl_table_lookup_8b_f32(size_t nsq, size_t ncode, const uint8_t *codes, const float *sim_table,
    float *dis, float dis0, size_t codes_size, size_t sim_table_size, size_t dis_size);

KRL_API_PUBLIC int krl_table_lookup_8b_f32_by_idx(size_t nsq, size_t ncode, const uint8_t *codes,
    const float *sim_table, float *dis, float dis0, const size_t *idx, size_t codes_size, size_t sim_table_size,
    size_t dis_size);

#ifdef __cplusplus
}
#endif

#endif  // KRL_H