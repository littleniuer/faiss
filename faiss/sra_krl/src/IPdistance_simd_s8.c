#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include <stdio.h>

KRL_IMPRECISE_FUNCTION_BEGIN
int32_t krl_inner_product_s8s32(const int8_t *x, const int8_t *__restrict y, const size_t d)
{
    size_t i;
    int32_t res;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 64;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 64 */
    for (i = 0; i + double_round <= d; i += double_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t x8_1 = vld1q_s8(x + i + 16);
        const int8x16_t x8_2 = vld1q_s8(x + i + 32);
        const int8x16_t x8_3 = vld1q_s8(x + i + 48);

        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + i + 16);
        const int8x16_t y8_2 = vld1q_s8(y + i + 32);
        const int8x16_t y8_3 = vld1q_s8(y + i + 48);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res2 = vdotq_s32(res2, x8_1, y8_1);
        res3 = vdotq_s32(res3, x8_2, y8_2);
        res4 = vdotq_s32(res4, x8_3, y8_3);
    }

    /* Process remaining vectors in batches of 16 */
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y + i);
        res1 = vdotq_s32(res1, x8_0, y8_0);
    }

    /* Sum the results */
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);
    res1 = vaddq_s32(res1, res3);
    res = vaddvq_s32(res1);

    /* Handle remaining elements */
    for (; i < d; i++) {
        res += (int32_t)(x[i]) * (int32_t)(y[i]);
    }

    return res;
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch2_s8f32(const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    int32x4_t res1 = vdupq_n_s32(0);
    int32x4_t res2 = vdupq_n_s32(0);
    int32x4_t res3 = vdupq_n_s32(0);
    int32x4_t res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 32 */
    for (i = 0; i + double_round <= d; i += double_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t x8_1 = vld1q_s8(x + i + 16);

        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + i + 16);
        const int8x16_t y8_2 = vld1q_s8(y + d + i);
        const int8x16_t y8_3 = vld1q_s8(y + d + i + 16);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res2 = vdotq_s32(res2, x8_1, y8_1);
        res3 = vdotq_s32(res3, x8_0, y8_2);
        res4 = vdotq_s32(res4, x8_1, y8_3);
    }

    /* Process remaining vectors in batches of 16 */
    for (; i + single_round <= d; i += single_round) {
        const int8x16_t x8_0 = vld1q_s8(x + i);
        const int8x16_t y8_0 = vld1q_s8(y + i);
        const int8x16_t y8_1 = vld1q_s8(y + d + i);

        res1 = vdotq_s32(res1, x8_0, y8_0);
        res3 = vdotq_s32(res3, x8_0, y8_1);
    }

    /* Sum the results */
    res1 = vaddq_s32(res1, res2);
    res3 = vaddq_s32(res3, res4);

    /* Store the results */
    dis[0] = (float)vaddvq_s32(res1);
    dis[1] = (float)vaddvq_s32(res3);

    /* Handle remaining elements */
    if (i < d) {
        dis[0] += (int32_t)(x[i]) * (int32_t)(y[i]);
        dis[1] += (int32_t)(x[i]) * (int32_t)(y[d + i]);
        for (i++; i < d; ++i) {
            dis[0] += (int32_t)(x[i]) * (int32_t)(y[i]);
            dis[1] += (int32_t)(x[i]) * (int32_t)(y[d + i]);
        }
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch4_s8f32(const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y + i);
        int8x16_t neon_base2 = vld1q_s8(y + d + i);
        int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
        int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);

    /* Store the results */
    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        float d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        float d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        float d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_inner_product_batch8_s8f32(const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);

    /* Process vectors in batches of 16 */
    for (i = 0; i + single_round <= d; i += single_round) {
        int8x16_t neon_query = vld1q_s8(x + i);
        int8x16_t neon_base1 = vld1q_s8(y + i);
        int8x16_t neon_base2 = vld1q_s8(y + d + i);
        int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
        int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
        int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
        int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
        int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
        int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

        neon_res1 = vdotq_s32(neon_res1, neon_query, neon_base1);
        neon_res2 = vdotq_s32(neon_res2, neon_query, neon_base2);
        neon_res3 = vdotq_s32(neon_res3, neon_query, neon_base3);
        neon_res4 = vdotq_s32(neon_res4, neon_query, neon_base4);
        neon_res5 = vdotq_s32(neon_res5, neon_query, neon_base5);
        neon_res6 = vdotq_s32(neon_res6, neon_query, neon_base6);
        neon_res7 = vdotq_s32(neon_res7, neon_query, neon_base7);
        neon_res8 = vdotq_s32(neon_res8, neon_query, neon_base8);
    }

    /* Sum the results */
    neon_res1 = vpaddq_s32(neon_res1, neon_res2);
    neon_res3 = vpaddq_s32(neon_res3, neon_res4);
    neon_res5 = vpaddq_s32(neon_res5, neon_res6);
    neon_res7 = vpaddq_s32(neon_res7, neon_res8);
    neon_res1 = vpaddq_s32(neon_res1, neon_res3);
    neon_res5 = vpaddq_s32(neon_res5, neon_res7);

    /* Store the results */
    vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
    vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        float d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        float d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        float d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        float d4 = (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
        float d5 = (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
        float d6 = (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
        float d7 = (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
            d4 += (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
            d5 += (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
            d6 += (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
            d7 += (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        }
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
static void krl_inner_product_prefetch_batch16_s8f32(
    const int8_t *x, const int8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;   /* 128 / 8 */
    constexpr size_t prefetch_round = 64; /* 4 * single_round */

    int32x4_t neon_res1 = vdupq_n_s32(0);
    int32x4_t neon_res2 = vdupq_n_s32(0);
    int32x4_t neon_res3 = vdupq_n_s32(0);
    int32x4_t neon_res4 = vdupq_n_s32(0);
    int32x4_t neon_res5 = vdupq_n_s32(0);
    int32x4_t neon_res6 = vdupq_n_s32(0);
    int32x4_t neon_res7 = vdupq_n_s32(0);
    int32x4_t neon_res8 = vdupq_n_s32(0);
    int32x4_t neon_res9 = vdupq_n_s32(0);
    int32x4_t neon_res10 = vdupq_n_s32(0);
    int32x4_t neon_res11 = vdupq_n_s32(0);
    int32x4_t neon_res12 = vdupq_n_s32(0);
    int32x4_t neon_res13 = vdupq_n_s32(0);
    int32x4_t neon_res14 = vdupq_n_s32(0);
    int32x4_t neon_res15 = vdupq_n_s32(0);
    int32x4_t neon_res16 = vdupq_n_s32(0);

    /* Process vectors with prefetching */
    if (d >= prefetch_round) {
        for (i = 0; i < d - prefetch_round; i += prefetch_round) {
            /* Prefetch data to L1 cache */
            prefetch_L1(x + i + prefetch_round);
            prefetch_Lx(y + i + prefetch_round);
            prefetch_Lx(y + d + i + prefetch_round);
            prefetch_Lx(y + 2 * d + i + prefetch_round);
            prefetch_Lx(y + 3 * d + i + prefetch_round);
            prefetch_Lx(y + 4 * d + i + prefetch_round);
            prefetch_Lx(y + 5 * d + i + prefetch_round);
            prefetch_Lx(y + 6 * d + i + prefetch_round);
            prefetch_Lx(y + 7 * d + i + prefetch_round);
            prefetch_Lx(y + 8 * d + i + prefetch_round);
            prefetch_Lx(y + 9 * d + i + prefetch_round);
            prefetch_Lx(y + 10 * d + i + prefetch_round);
            prefetch_Lx(y + 11 * d + i + prefetch_round);
            prefetch_Lx(y + 12 * d + i + prefetch_round);
            prefetch_Lx(y + 13 * d + i + prefetch_round);
            prefetch_Lx(y + 14 * d + i + prefetch_round);
            prefetch_Lx(y + 15 * d + i + prefetch_round);

            /* Process data in single_round chunks */
            for (size_t j = 0; j < prefetch_round; j += single_round) {
                int8x16_t neon_query = vld1q_s8(x + i + j);
                int8x16_t neon_base1 = vld1q_s8(y + i + j);
                int8x16_t neon_base2 = vld1q_s8(y + d + i + j);
                int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i + j);
                int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i + j);
                int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i + j);
                int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i + j);
                int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i + j);
                int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i + j);

                neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
                neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
                neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
                neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
                neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
                neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
                neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
                neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

                neon_base1 = vld1q_s8(y + 8 * d + i + j);
                neon_base2 = vld1q_s8(y + 9 * d + i + j);
                neon_base3 = vld1q_s8(y + 10 * d + i + j);
                neon_base4 = vld1q_s8(y + 11 * d + i + j);
                neon_base5 = vld1q_s8(y + 12 * d + i + j);
                neon_base6 = vld1q_s8(y + 13 * d + i + j);
                neon_base7 = vld1q_s8(y + 14 * d + i + j);
                neon_base8 = vld1q_s8(y + 15 * d + i + j);

                neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
                neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
                neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
                neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
                neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
                neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
                neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
                neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
            }
        }

        /* Process remaining data */
        for (; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y + i);
            int8x16_t neon_base2 = vld1q_s8(y + d + i);
            int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
            int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
            int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
            int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
            int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
            int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y + 8 * d + i);
            neon_base2 = vld1q_s8(y + 9 * d + i);
            neon_base3 = vld1q_s8(y + 10 * d + i);
            neon_base4 = vld1q_s8(y + 11 * d + i);
            neon_base5 = vld1q_s8(y + 12 * d + i);
            neon_base6 = vld1q_s8(y + 13 * d + i);
            neon_base7 = vld1q_s8(y + 14 * d + i);
            neon_base8 = vld1q_s8(y + 15 * d + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);
        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);

        vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));
    } else {
        /* Process without prefetching */
        for (i = 0; i + single_round <= d; i += single_round) {
            int8x16_t neon_query = vld1q_s8(x + i);
            int8x16_t neon_base1 = vld1q_s8(y + i);
            int8x16_t neon_base2 = vld1q_s8(y + d + i);
            int8x16_t neon_base3 = vld1q_s8(y + 2 * d + i);
            int8x16_t neon_base4 = vld1q_s8(y + 3 * d + i);
            int8x16_t neon_base5 = vld1q_s8(y + 4 * d + i);
            int8x16_t neon_base6 = vld1q_s8(y + 5 * d + i);
            int8x16_t neon_base7 = vld1q_s8(y + 6 * d + i);
            int8x16_t neon_base8 = vld1q_s8(y + 7 * d + i);

            neon_res1 = vdotq_s32(neon_res1, neon_base1, neon_query);
            neon_res2 = vdotq_s32(neon_res2, neon_base2, neon_query);
            neon_res3 = vdotq_s32(neon_res3, neon_base3, neon_query);
            neon_res4 = vdotq_s32(neon_res4, neon_base4, neon_query);
            neon_res5 = vdotq_s32(neon_res5, neon_base5, neon_query);
            neon_res6 = vdotq_s32(neon_res6, neon_base6, neon_query);
            neon_res7 = vdotq_s32(neon_res7, neon_base7, neon_query);
            neon_res8 = vdotq_s32(neon_res8, neon_base8, neon_query);

            neon_base1 = vld1q_s8(y + 8 * d + i);
            neon_base2 = vld1q_s8(y + 9 * d + i);
            neon_base3 = vld1q_s8(y + 10 * d + i);
            neon_base4 = vld1q_s8(y + 11 * d + i);
            neon_base5 = vld1q_s8(y + 12 * d + i);
            neon_base6 = vld1q_s8(y + 13 * d + i);
            neon_base7 = vld1q_s8(y + 14 * d + i);
            neon_base8 = vld1q_s8(y + 15 * d + i);

            neon_res9 = vdotq_s32(neon_res9, neon_base1, neon_query);
            neon_res10 = vdotq_s32(neon_res10, neon_base2, neon_query);
            neon_res11 = vdotq_s32(neon_res11, neon_base3, neon_query);
            neon_res12 = vdotq_s32(neon_res12, neon_base4, neon_query);
            neon_res13 = vdotq_s32(neon_res13, neon_base5, neon_query);
            neon_res14 = vdotq_s32(neon_res14, neon_base6, neon_query);
            neon_res15 = vdotq_s32(neon_res15, neon_base7, neon_query);
            neon_res16 = vdotq_s32(neon_res16, neon_base8, neon_query);
        }
        /* Sum the results */
        neon_res1 = vpaddq_s32(neon_res1, neon_res2);
        neon_res3 = vpaddq_s32(neon_res3, neon_res4);
        neon_res5 = vpaddq_s32(neon_res5, neon_res6);
        neon_res7 = vpaddq_s32(neon_res7, neon_res8);
        neon_res9 = vpaddq_s32(neon_res9, neon_res10);
        neon_res11 = vpaddq_s32(neon_res11, neon_res12);
        neon_res13 = vpaddq_s32(neon_res13, neon_res14);
        neon_res15 = vpaddq_s32(neon_res15, neon_res16);
        neon_res1 = vpaddq_s32(neon_res1, neon_res3);
        neon_res5 = vpaddq_s32(neon_res5, neon_res7);
        neon_res9 = vpaddq_s32(neon_res9, neon_res11);
        neon_res13 = vpaddq_s32(neon_res13, neon_res15);

        /* Store the results */
        vst1q_f32(dis, vcvtq_f32_s32(neon_res1));
        vst1q_f32(dis + 4, vcvtq_f32_s32(neon_res5));
        vst1q_f32(dis + 8, vcvtq_f32_s32(neon_res9));
        vst1q_f32(dis + 12, vcvtq_f32_s32(neon_res13));
    }

    /* Handle remaining elements */
    if (i < d) {
        float d0 = (int32_t)(x[i]) * (int32_t) * (y + i);
        float d1 = (int32_t)(x[i]) * (int32_t) * (y + d + i);
        float d2 = (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
        float d3 = (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
        float d4 = (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
        float d5 = (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
        float d6 = (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
        float d7 = (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
        float d8 = (int32_t)(x[i]) * (int32_t) * (y + 8 * d + i);
        float d9 = (int32_t)(x[i]) * (int32_t) * (y + 9 * d + i);
        float d10 = (int32_t)(x[i]) * (int32_t) * (y + 10 * d + i);
        float d11 = (int32_t)(x[i]) * (int32_t) * (y + 11 * d + i);
        float d12 = (int32_t)(x[i]) * (int32_t) * (y + 12 * d + i);
        float d13 = (int32_t)(x[i]) * (int32_t) * (y + 13 * d + i);
        float d14 = (int32_t)(x[i]) * (int32_t) * (y + 14 * d + i);
        float d15 = (int32_t)(x[i]) * (int32_t) * (y + 15 * d + i);
        for (i++; i < d; ++i) {
            d0 += (int32_t)(x[i]) * (int32_t) * (y + i);
            d1 += (int32_t)(x[i]) * (int32_t) * (y + d + i);
            d2 += (int32_t)(x[i]) * (int32_t) * (y + 2 * d + i);
            d3 += (int32_t)(x[i]) * (int32_t) * (y + 3 * d + i);
            d4 += (int32_t)(x[i]) * (int32_t) * (y + 4 * d + i);
            d5 += (int32_t)(x[i]) * (int32_t) * (y + 5 * d + i);
            d6 += (int32_t)(x[i]) * (int32_t) * (y + 6 * d + i);
            d7 += (int32_t)(x[i]) * (int32_t) * (y + 7 * d + i);
            d8 += (int32_t)(x[i]) * (int32_t) * (y + 8 * d + i);
            d9 += (int32_t)(x[i]) * (int32_t) * (y + 9 * d + i);
            d10 += (int32_t)(x[i]) * (int32_t) * (y + 10 * d + i);
            d11 += (int32_t)(x[i]) * (int32_t) * (y + 11 * d + i);
            d12 += (int32_t)(x[i]) * (int32_t) * (y + 12 * d + i);
            d13 += (int32_t)(x[i]) * (int32_t) * (y + 13 * d + i);
            d14 += (int32_t)(x[i]) * (int32_t) * (y + 14 * d + i);
            d15 += (int32_t)(x[i]) * (int32_t) * (y + 15 * d + i);
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
    }
}
KRL_IMPRECISE_FUNCTION_END

int krl_inner_product_ny_s8f32(float *dis, const int8_t *x, const int8_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    /* Process vectors in batches of 16 */
    for (; i + 16 <= ny; i += 16) {
        krl_inner_product_prefetch_batch16_s8f32(x, y + i * d, d, dis + i);
    }

    /* Handle remaining vectors */
    if (ny & 8) {
        krl_inner_product_batch8_s8f32(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_inner_product_batch4_s8f32(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_inner_product_batch2_s8f32(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        dis[i] = (float)krl_inner_product_s8s32(x, y + i * d, d);
    }
    return SUCCESS;
}