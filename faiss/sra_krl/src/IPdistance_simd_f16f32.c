#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include <stdio.h>

KRL_IMPRECISE_FUNCTION_BEGIN
float krl_inner_product_f16f32(const uint16_t *u16_x, const uint16_t *__restrict u16_y, const size_t d)
{
    const float16_t *x = (const float16_t *)u16_x;
    const float16_t *y = (const float16_t *)u16_y;
    size_t i;
    float res;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 32;

    /* Initialize result registers */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);

    /* Main computation loop with double rounds */
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);
        float16x8_t x8_2 = vld1q_f16(x + i + 16);
        float16x8_t x8_3 = vld1q_f16(x + i + 24);

        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + i + 16);
        float16x8_t y8_3 = vld1q_f16(y + i + 24);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_low_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_low_f16(res3, x8_2, y8_2);
        res4 = vfmlalq_low_f16(res4, x8_3, y8_3);

        res1 = vfmlalq_high_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_high_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_high_f16(res3, x8_2, y8_2);
        res4 = vfmlalq_high_f16(res4, x8_3, y8_3);
    }

    /* Handle remaining elements with single rounds */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res3 = vfmlalq_high_f16(res3, x8_0, y8_0);
    }

    /* Accumulate results */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    res1 = vaddq_f32(res1, res3);
    res = vaddvq_f32(res1);

    /* Handle remaining elements */
    for (; i < d; i++) {
        res += (float)(x[i] * y[i]);
    }

    return res;
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch2_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;  /* Number of elements processed in a single-round loop */
    constexpr size_t double_round = 16; /* Number of elements processed in a double-round loop */

    /* Initialize result registers */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);

    /* Process data in double-round chunks */
    for (i = 0; i + double_round <= d; i += double_round) {
        /* Load data for the first part */
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        /* Load data for the second part */
        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + d + i);
        float16x8_t y8_3 = vld1q_f16(y + d + i + 8);

        /* Perform vectorized multiplication and accumulation */
        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_low_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_low_f16(res3, x8_0, y8_2);
        res4 = vfmlalq_low_f16(res4, x8_1, y8_3);

        res1 = vfmlalq_high_f16(res1, x8_0, y8_0);
        res2 = vfmlalq_high_f16(res2, x8_1, y8_1);
        res3 = vfmlalq_high_f16(res3, x8_0, y8_2);
        res4 = vfmlalq_high_f16(res4, x8_1, y8_3);
    }

    /* Process remaining data in single-round chunks */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + d + i);

        res1 = vfmlalq_low_f16(res1, x8_0, y8_0);
        res3 = vfmlalq_low_f16(res3, x8_0, y8_1);
        res2 = vfmlalq_high_f16(res2, x8_0, y8_0);
        res4 = vfmlalq_high_f16(res4, x8_0, y8_1);
    }

    /* Combine results from the four registers */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);

    /* Store the final results */
    dis[0] = vaddvq_f32(res1);
    dis[1] = vaddvq_f32(res3);

    /* Handle any remaining elements */
    for (; i < d; i++) {
        dis[0] += (float)(x[i] * y[i]);
        dis[1] += (float)(x[i] * y[i + d]);
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch4_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    const size_t single_round = 8; /* Number of elements processed in a single-round loop */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);

    /* Process data in single-round chunks */
    for (i = 0; i + single_round <= d; i += single_round) {
        /* Load data for the query and four bases */
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y + i);
        float16x8_t neon_base2 = vld1q_f16(y + d + i);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);

        /* Perform vectorized multiplication and accumulation */
        neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
    }

    /* Store the final results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);

    /* Handle any remaining elements */
    if (i < d) {
        /* Initialize partial sums */
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);

        /* Accumulate remaining elements */
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
        }

        /* Add partial sums to the final results */
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch8_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed in a single-round loop */

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);

    /* Process data in single-round chunks */
    for (i = 0; i + single_round <= d; i += single_round) {
        /* Load data for the query and eight bases */
        float16x8_t neon_query = vld1q_f16(x + i);
        float16x8_t neon_base1 = vld1q_f16(y + i);
        float16x8_t neon_base2 = vld1q_f16(y + d + i);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

        /* Perform vectorized multiplication and accumulation */
        neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);
    }

    /* Store the final results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);
    dis[4] = vaddvq_f32(neon_res5);
    dis[5] = vaddvq_f32(neon_res6);
    dis[6] = vaddvq_f32(neon_res7);
    dis[7] = vaddvq_f32(neon_res8);

    /* Handle any remaining elements */
    if (i < d) {
        /* Initialize partial sums */
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);

        /* Accumulate remaining elements */
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
        }

        /* Add partial sums to the final results */
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_prefetch_batch16_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t multi_round = 32;

    /* Initialize result registers */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);
    float32x4_t neon_res9 = vdupq_n_f32(0.0f);
    float32x4_t neon_res10 = vdupq_n_f32(0.0f);
    float32x4_t neon_res11 = vdupq_n_f32(0.0f);
    float32x4_t neon_res12 = vdupq_n_f32(0.0f);
    float32x4_t neon_res13 = vdupq_n_f32(0.0f);
    float32x4_t neon_res14 = vdupq_n_f32(0.0f);
    float32x4_t neon_res15 = vdupq_n_f32(0.0f);
    float32x4_t neon_res16 = vdupq_n_f32(0.0f);

    if (d >= multi_round) {
        /* Process data in multi-round chunks with prefetching */
        for (i = 0; i < d - multi_round; i += multi_round) {
            /* Prefetch data for the next multi_round */
            prefetch_L1(x + i + multi_round);
            prefetch_Lx(y + i + multi_round);
            prefetch_Lx(y + d + i + multi_round);
            prefetch_Lx(y + 2 * d + i + multi_round);
            prefetch_Lx(y + 3 * d + i + multi_round);
            prefetch_Lx(y + 4 * d + i + multi_round);
            prefetch_Lx(y + 5 * d + i + multi_round);
            prefetch_Lx(y + 6 * d + i + multi_round);
            prefetch_Lx(y + 7 * d + i + multi_round);
            prefetch_Lx(y + 8 * d + i + multi_round);
            prefetch_Lx(y + 9 * d + i + multi_round);
            prefetch_Lx(y + 10 * d + i + multi_round);
            prefetch_Lx(y + 11 * d + i + multi_round);
            prefetch_Lx(y + 12 * d + i + multi_round);
            prefetch_Lx(y + 13 * d + i + multi_round);
            prefetch_Lx(y + 14 * d + i + multi_round);
            prefetch_Lx(y + 15 * d + i + multi_round);

            /* Process data in single_round chunks */
            for (size_t j = 0; j < multi_round; j += single_round) {
                /* Load data for the query and eight bases */
                const float16x8_t neon_query = vld1q_f16(x + i + j);
                float16x8_t neon_base1 = vld1q_f16(y + i + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + i + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i + j);
                float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i + j);
                float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i + j);
                float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i + j);
                float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i + j);

                /* Perform vectorized multiplication and accumulation */
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

                neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

                /* Load data for the next eight bases */
                neon_base1 = vld1q_f16(y + 8 * d + i + j);
                neon_base2 = vld1q_f16(y + 9 * d + i + j);
                neon_base3 = vld1q_f16(y + 10 * d + i + j);
                neon_base4 = vld1q_f16(y + 11 * d + i + j);
                neon_base5 = vld1q_f16(y + 12 * d + i + j);
                neon_base6 = vld1q_f16(y + 13 * d + i + j);
                neon_base7 = vld1q_f16(y + 14 * d + i + j);
                neon_base8 = vld1q_f16(y + 15 * d + i + j);

                /* Perform vectorized multiplication and accumulation */
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

                neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
            }
        }

        /* Process remaining data in single_round chunks */
        for (; i + single_round <= d; i += single_round) {
            /* Load data for the query and eight bases */
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

            /* Load data for the next eight bases */
            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
        }
    } else {
        /* Process data in single_round chunks without prefetching */
        for (i = 0; i + single_round <= d; i += single_round) {
            /* Load data for the query and eight bases */
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_query, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_query, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_query, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_query, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_query, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_query, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_query, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_query, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_query, neon_base8);

            /* Load data for the next eight bases */
            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            /* Perform vectorized multiplication and accumulation */
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_query, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_query, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_query, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_query, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_query, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_query, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_query, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_query, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_query, neon_base8);
        }
    }

    /* Store the final results */
    dis[0] = vaddvq_f32(neon_res1);
    dis[1] = vaddvq_f32(neon_res2);
    dis[2] = vaddvq_f32(neon_res3);
    dis[3] = vaddvq_f32(neon_res4);
    dis[4] = vaddvq_f32(neon_res5);
    dis[5] = vaddvq_f32(neon_res6);
    dis[6] = vaddvq_f32(neon_res7);
    dis[7] = vaddvq_f32(neon_res8);
    dis[8] = vaddvq_f32(neon_res9);
    dis[9] = vaddvq_f32(neon_res10);
    dis[10] = vaddvq_f32(neon_res11);
    dis[11] = vaddvq_f32(neon_res12);
    dis[12] = vaddvq_f32(neon_res13);
    dis[13] = vaddvq_f32(neon_res14);
    dis[14] = vaddvq_f32(neon_res15);
    dis[15] = vaddvq_f32(neon_res16);

    /* Handle any remaining elements */
    if (i < d) {
        /* Initialize partial sums */
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);
        float d8 = x[i] * *(y + 8 * d + i);
        float d9 = x[i] * *(y + 9 * d + i);
        float d10 = x[i] * *(y + 10 * d + i);
        float d11 = x[i] * *(y + 11 * d + i);
        float d12 = x[i] * *(y + 12 * d + i);
        float d13 = x[i] * *(y + 13 * d + i);
        float d14 = x[i] * *(y + 14 * d + i);
        float d15 = x[i] * *(y + 15 * d + i);

        /* Accumulate remaining elements */
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
            d8 += x[i] * *(y + 8 * d + i);
            d9 += x[i] * *(y + 9 * d + i);
            d10 += x[i] * *(y + 10 * d + i);
            d11 += x[i] * *(y + 11 * d + i);
            d12 += x[i] * *(y + 12 * d + i);
            d13 += x[i] * *(y + 13 * d + i);
            d14 += x[i] * *(y + 14 * d + i);
            d15 += x[i] * *(y + 15 * d + i);
        }

        /* Add partial sums to the final results */
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
    }
}
KRL_IMPRECISE_FUNCTION_END

/*
 * @brief Compute the inner product of sixteen batches of half-precision floating-point vectors.
 * @param x Pointer to the input vector (half-precision float).
 * @param y Pointer to the input vector (half-precision float), which contains sixteen batches.
 * @param d The length of the vectors.
 * @param dis Pointer to the output array where the results are stored.
 */
KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_prefetch_batch24_f16f32(
    const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* Number of elements processed per round (128-bit NEON vector) */
    constexpr size_t multi_round = 32; /* Number of elements processed per multi-round (4 NEON vectors) */

    /* Initialize NEON registers for accumulating results */
    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);
    float32x4_t neon_res9 = vdupq_n_f32(0.0f);
    float32x4_t neon_res10 = vdupq_n_f32(0.0f);
    float32x4_t neon_res11 = vdupq_n_f32(0.0f);
    float32x4_t neon_res12 = vdupq_n_f32(0.0f);
    float32x4_t neon_res13 = vdupq_n_f32(0.0f);
    float32x4_t neon_res14 = vdupq_n_f32(0.0f);
    float32x4_t neon_res15 = vdupq_n_f32(0.0f);
    float32x4_t neon_res16 = vdupq_n_f32(0.0f);
    float32x4_t neon_res17 = vdupq_n_f32(0.0f);
    float32x4_t neon_res18 = vdupq_n_f32(0.0f);
    float32x4_t neon_res19 = vdupq_n_f32(0.0f);
    float32x4_t neon_res20 = vdupq_n_f32(0.0f);
    float32x4_t neon_res21 = vdupq_n_f32(0.0f);
    float32x4_t neon_res22 = vdupq_n_f32(0.0f);
    float32x4_t neon_res23 = vdupq_n_f32(0.0f);
    float32x4_t neon_res24 = vdupq_n_f32(0.0f);

    if (d >= multi_round) {
        for (i = 0; i < d - multi_round; i += multi_round) {
            const size_t next_i = i + multi_round;
            /* Prefetch data to improve cache performance */
            prefetch_L1(x + next_i);
            prefetch_Lx(y + next_i);
            prefetch_Lx(y + d + next_i);
            prefetch_Lx(y + 2 * d + next_i);
            prefetch_Lx(y + 3 * d + next_i);
            prefetch_Lx(y + 4 * d + next_i);
            prefetch_Lx(y + 5 * d + next_i);
            prefetch_Lx(y + 6 * d + next_i);
            prefetch_Lx(y + 7 * d + next_i);
            prefetch_Lx(y + 8 * d + next_i);
            prefetch_Lx(y + 9 * d + next_i);
            prefetch_Lx(y + 10 * d + next_i);
            prefetch_Lx(y + 11 * d + next_i);
            prefetch_Lx(y + 12 * d + next_i);
            prefetch_Lx(y + 13 * d + next_i);
            prefetch_Lx(y + 14 * d + next_i);
            prefetch_Lx(y + 15 * d + next_i);
            prefetch_Lx(y + 16 * d + next_i);
            prefetch_Lx(y + 17 * d + next_i);
            prefetch_Lx(y + 18 * d + next_i);
            prefetch_Lx(y + 19 * d + next_i);
            prefetch_Lx(y + 20 * d + next_i);
            prefetch_Lx(y + 21 * d + next_i);
            prefetch_Lx(y + 22 * d + next_i);
            prefetch_Lx(y + 23 * d + next_i);

            for (size_t j = i; j < next_i; j += single_round) {
                /* Load query vector */
                const float16x8_t neon_query = vld1q_f16(x + j);
                /* Load base vectors and compute inner products */
                float16x8_t neon_base1 = vld1q_f16(y + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + j);
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 4 * d + j);
                neon_base2 = vld1q_f16(y + 5 * d + j);
                neon_base3 = vld1q_f16(y + 6 * d + j);
                neon_base4 = vld1q_f16(y + 7 * d + j);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 8 * d + j);
                neon_base2 = vld1q_f16(y + 9 * d + j);
                neon_base3 = vld1q_f16(y + 10 * d + j);
                neon_base4 = vld1q_f16(y + 11 * d + j);
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 12 * d + j);
                neon_base2 = vld1q_f16(y + 13 * d + j);
                neon_base3 = vld1q_f16(y + 14 * d + j);
                neon_base4 = vld1q_f16(y + 15 * d + j);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 16 * d + j);
                neon_base2 = vld1q_f16(y + 17 * d + j);
                neon_base3 = vld1q_f16(y + 18 * d + j);
                neon_base4 = vld1q_f16(y + 19 * d + j);
                neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
                neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
                neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
                neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
                neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

                neon_base1 = vld1q_f16(y + 20 * d + j);
                neon_base2 = vld1q_f16(y + 21 * d + j);
                neon_base3 = vld1q_f16(y + 22 * d + j);
                neon_base4 = vld1q_f16(y + 23 * d + j);
                neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
                neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
                neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
                neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
                neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
            }
        }

        for (; i <= d - single_round; i += single_round) {
            /* Load query vector */
            const float16x8_t neon_query = vld1q_f16(x + i);
            /* Load base vectors and compute inner products */
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_query);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_query);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_query);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_query);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_query);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_query);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_query);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_query);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_query);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_query);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_query);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_query);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_query);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_query);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_query);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_query);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_query);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_query);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_query);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_query);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_query);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_query);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_query);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_query);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_query);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_query);
        }
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        dis[8] = vaddvq_f32(neon_res9);
        dis[9] = vaddvq_f32(neon_res10);
        dis[10] = vaddvq_f32(neon_res11);
        dis[11] = vaddvq_f32(neon_res12);
        dis[12] = vaddvq_f32(neon_res13);
        dis[13] = vaddvq_f32(neon_res14);
        dis[14] = vaddvq_f32(neon_res15);
        dis[15] = vaddvq_f32(neon_res16);
        dis[16] = vaddvq_f32(neon_res17);
        dis[17] = vaddvq_f32(neon_res18);
        dis[18] = vaddvq_f32(neon_res19);
        dis[19] = vaddvq_f32(neon_res20);
        dis[20] = vaddvq_f32(neon_res21);
        dis[21] = vaddvq_f32(neon_res22);
        dis[22] = vaddvq_f32(neon_res23);
        dis[23] = vaddvq_f32(neon_res24);
    } else {
        for (int i = 0; i < 24; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float d0 = x[i] * *(y + i);
        float d1 = x[i] * *(y + d + i);
        float d2 = x[i] * *(y + 2 * d + i);
        float d3 = x[i] * *(y + 3 * d + i);
        float d4 = x[i] * *(y + 4 * d + i);
        float d5 = x[i] * *(y + 5 * d + i);
        float d6 = x[i] * *(y + 6 * d + i);
        float d7 = x[i] * *(y + 7 * d + i);
        float d8 = x[i] * *(y + 8 * d + i);
        float d9 = x[i] * *(y + 9 * d + i);
        float d10 = x[i] * *(y + 10 * d + i);
        float d11 = x[i] * *(y + 11 * d + i);
        float d12 = x[i] * *(y + 12 * d + i);
        float d13 = x[i] * *(y + 13 * d + i);
        float d14 = x[i] * *(y + 14 * d + i);
        float d15 = x[i] * *(y + 15 * d + i);
        float d16 = x[i] * *(y + 16 * d + i);
        float d17 = x[i] * *(y + 17 * d + i);
        float d18 = x[i] * *(y + 18 * d + i);
        float d19 = x[i] * *(y + 19 * d + i);
        float d20 = x[i] * *(y + 20 * d + i);
        float d21 = x[i] * *(y + 21 * d + i);
        float d22 = x[i] * *(y + 22 * d + i);
        float d23 = x[i] * *(y + 23 * d + i);
        for (i++; i < d; ++i) {
            d0 += x[i] * *(y + i);
            d1 += x[i] * *(y + d + i);
            d2 += x[i] * *(y + 2 * d + i);
            d3 += x[i] * *(y + 3 * d + i);
            d4 += x[i] * *(y + 4 * d + i);
            d5 += x[i] * *(y + 5 * d + i);
            d6 += x[i] * *(y + 6 * d + i);
            d7 += x[i] * *(y + 7 * d + i);
            d8 += x[i] * *(y + 8 * d + i);
            d9 += x[i] * *(y + 9 * d + i);
            d10 += x[i] * *(y + 10 * d + i);
            d11 += x[i] * *(y + 11 * d + i);
            d12 += x[i] * *(y + 12 * d + i);
            d13 += x[i] * *(y + 13 * d + i);
            d14 += x[i] * *(y + 14 * d + i);
            d15 += x[i] * *(y + 15 * d + i);
            d16 += x[i] * *(y + 16 * d + i);
            d17 += x[i] * *(y + 17 * d + i);
            d18 += x[i] * *(y + 18 * d + i);
            d19 += x[i] * *(y + 19 * d + i);
            d20 += x[i] * *(y + 20 * d + i);
            d21 += x[i] * *(y + 21 * d + i);
            d22 += x[i] * *(y + 22 * d + i);
            d23 += x[i] * *(y + 23 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
        dis[4] += d4;
        dis[5] += d5;
        dis[6] += d6;
        dis[7] += d7;
        dis[8] += d8;
        dis[9] += d9;
        dis[10] += d10;
        dis[11] += d11;
        dis[12] += d12;
        dis[13] += d13;
        dis[14] += d14;
        dis[15] += d15;
        dis[16] += d16;
        dis[17] += d17;
        dis[18] += d18;
        dis[19] += d19;
        dis[20] += d20;
        dis[21] += d21;
        dis[22] += d22;
        dis[23] += d23;
    }
}
KRL_IMPRECISE_FUNCTION_END

int krl_inner_product_ny_f16f32(float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    /* Process vectors in batches of 24 */
    for (; i + 24 <= ny; i += 24) {
        /* Prefetch query vector */
        prefetch_L1(x);
        /* Prefetch database vectors */
        prefetch_Lx(y + i * d);
        /* Compute inner products for 24 vectors */
        krl_inner_product_prefetch_batch24_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
    }

    /* Handle remaining vectors in batches of 16 */
    if (i + 16 <= ny) {
        /* Prefetch query vector */
        prefetch_L1(x);
        /* Prefetch database vectors */
        prefetch_Lx(y + i * d);
        /* Compute inner products for 16 vectors */
        krl_inner_product_prefetch_batch16_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        /* Handle remaining vectors in batches of 8 */
        krl_inner_product_batch8_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 8;
    }

    /* Handle remaining vectors in batches of 4 */
    if (ny & 4) {
        krl_inner_product_batch4_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 4;
    }

    /* Handle remaining vectors in batches of 2 */
    if (ny & 2) {
        krl_inner_product_batch2_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 2;
    }

    /* Handle the last remaining vector */
    if (ny & 1) {
        dis[i] = krl_inner_product_f16f32(x, y + i * d, d);
    }
    return SUCCESS;
}