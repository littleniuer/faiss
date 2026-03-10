#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include <stdio.h>

KRL_IMPRECISE_FUNCTION_BEGIN
int krl_L2sqr_f16f32(
    const uint16_t *u16_x, const uint16_t *__restrict u16_y, const size_t d, float *dis, size_t dis_size)
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

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);
        float16x8_t d8_2 = vsubq_f16(x8_2, y8_2);
        float16x8_t d8_3 = vsubq_f16(x8_3, y8_3);

        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_low_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_low_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_low_f16(res4, d8_3, d8_3);

        res1 = vfmlalq_high_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_high_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_high_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_high_f16(res4, d8_3, d8_3);
    }

    /* Handle remaining elements with single rounds */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res3 = vfmlalq_high_f16(res3, d8_0, d8_0);
    }
    /* Accumulate results */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    res1 = vaddq_f32(res1, res3);
    res = vaddvq_f32(res1);
    /* Handle remaining elements */
    for (; i < d; i++) {
        const float16_t tmp = x[i] - y[i];
        res += (float)(tmp * tmp);
    }
    *dis = res;
    return SUCCESS;
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch2_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);
    for (i = 0; i + double_round <= d; i += double_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);
        float16x8_t y8_2 = vld1q_f16(y + d + i);
        float16x8_t y8_3 = vld1q_f16(y + d + i + 8);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);
        float16x8_t d8_2 = vsubq_f16(x8_0, y8_2);
        float16x8_t d8_3 = vsubq_f16(x8_1, y8_3);

        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_low_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_low_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_low_f16(res4, d8_3, d8_3);

        res1 = vfmlalq_high_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_high_f16(res2, d8_1, d8_1);
        res3 = vfmlalq_high_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_high_f16(res4, d8_3, d8_3);
    }
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + d + i);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0);
        float16x8_t d8_1 = vsubq_f16(x8_0, y8_1);
        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res3 = vfmlalq_low_f16(res3, d8_1, d8_1);
        res2 = vfmlalq_high_f16(res2, d8_0, d8_0);
        res4 = vfmlalq_high_f16(res4, d8_1, d8_1);
    }
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    dis[0] = vaddvq_f32(res1);
    dis[1] = vaddvq_f32(res3);
    for (; i < d; i++) {
        const float16_t tmp0 = x[i] - y[i];
        const float16_t tmp1 = x[i] - y[i + d];
        dis[0] += (float)(tmp0 * tmp0);
        dis[1] += (float)(tmp1 * tmp1);
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch4_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;

    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    if (likely(d >= double_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_query = vld1q_f16(x + single_round);
        neon_base1 = vld1q_f16(y + single_round);
        neon_base2 = vld1q_f16(y + d + single_round);
        neon_base3 = vld1q_f16(y + 2 * d + single_round);
        neon_base4 = vld1q_f16(y + 3 * d + single_round);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vsubq_f16(neon_base1, neon_query);
            neon_diff2 = vsubq_f16(neon_base2, neon_query);
            neon_diff3 = vsubq_f16(neon_base3, neon_query);
            neon_diff4 = vsubq_f16(neon_base4, neon_query);

            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        }
        neon_diff1 = vsubq_f16(neon_base1, neon_query);
        neon_diff2 = vsubq_f16(neon_base2, neon_query);
        neon_diff3 = vsubq_f16(neon_base3, neon_query);
        neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);

        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        i = single_round;
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
        }
        dis[0] += d0;
        dis[1] += d1;
        dis[2] += d2;
        dis[3] += d3;
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch8_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 16;

    float32x4_t neon_res1 = vdupq_n_f32(0.0f);
    float32x4_t neon_res2 = vdupq_n_f32(0.0f);
    float32x4_t neon_res3 = vdupq_n_f32(0.0f);
    float32x4_t neon_res4 = vdupq_n_f32(0.0f);
    float32x4_t neon_res5 = vdupq_n_f32(0.0f);
    float32x4_t neon_res6 = vdupq_n_f32(0.0f);
    float32x4_t neon_res7 = vdupq_n_f32(0.0f);
    float32x4_t neon_res8 = vdupq_n_f32(0.0f);

    if (likely(d >= double_round)) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_diff5 = vsubq_f16(neon_base5, neon_query);
        float16x8_t neon_diff6 = vsubq_f16(neon_base6, neon_query);
        float16x8_t neon_diff7 = vsubq_f16(neon_base7, neon_query);
        float16x8_t neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_query = vld1q_f16(x + single_round);
        neon_base1 = vld1q_f16(y + single_round);
        neon_base2 = vld1q_f16(y + d + single_round);
        neon_base3 = vld1q_f16(y + 2 * d + single_round);
        neon_base4 = vld1q_f16(y + 3 * d + single_round);
        neon_base5 = vld1q_f16(y + 4 * d + single_round);
        neon_base6 = vld1q_f16(y + 5 * d + single_round);
        neon_base7 = vld1q_f16(y + 6 * d + single_round);
        neon_base8 = vld1q_f16(y + 7 * d + single_round);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);

        for (i = double_round; i <= d - single_round; i += single_round) {
            neon_diff1 = vsubq_f16(neon_base1, neon_query);
            neon_diff2 = vsubq_f16(neon_base2, neon_query);
            neon_diff3 = vsubq_f16(neon_base3, neon_query);
            neon_diff4 = vsubq_f16(neon_base4, neon_query);
            neon_diff5 = vsubq_f16(neon_base5, neon_query);
            neon_diff6 = vsubq_f16(neon_base6, neon_query);
            neon_diff7 = vsubq_f16(neon_base7, neon_query);
            neon_diff8 = vsubq_f16(neon_base8, neon_query);

            neon_query = vld1q_f16(x + i);
            neon_base1 = vld1q_f16(y + i);
            neon_base2 = vld1q_f16(y + d + i);
            neon_base3 = vld1q_f16(y + 2 * d + i);
            neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base5 = vld1q_f16(y + 4 * d + i);
            neon_base6 = vld1q_f16(y + 5 * d + i);
            neon_base7 = vld1q_f16(y + 6 * d + i);
            neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        }
        neon_diff1 = vsubq_f16(neon_base1, neon_query);
        neon_diff2 = vsubq_f16(neon_base2, neon_query);
        neon_diff3 = vsubq_f16(neon_base3, neon_query);
        neon_diff4 = vsubq_f16(neon_base4, neon_query);
        neon_diff5 = vsubq_f16(neon_base5, neon_query);
        neon_diff6 = vsubq_f16(neon_base6, neon_query);
        neon_diff7 = vsubq_f16(neon_base7, neon_query);
        neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
    } else if (d >= single_round) {
        float16x8_t neon_query = vld1q_f16(x);
        float16x8_t neon_base1 = vld1q_f16(y);
        float16x8_t neon_base2 = vld1q_f16(y + d);
        float16x8_t neon_base3 = vld1q_f16(y + 2 * d);
        float16x8_t neon_base4 = vld1q_f16(y + 3 * d);
        float16x8_t neon_base5 = vld1q_f16(y + 4 * d);
        float16x8_t neon_base6 = vld1q_f16(y + 5 * d);
        float16x8_t neon_base7 = vld1q_f16(y + 6 * d);
        float16x8_t neon_base8 = vld1q_f16(y + 7 * d);

        float16x8_t neon_diff1 = vsubq_f16(neon_base1, neon_query);
        float16x8_t neon_diff2 = vsubq_f16(neon_base2, neon_query);
        float16x8_t neon_diff3 = vsubq_f16(neon_base3, neon_query);
        float16x8_t neon_diff4 = vsubq_f16(neon_base4, neon_query);
        float16x8_t neon_diff5 = vsubq_f16(neon_base5, neon_query);
        float16x8_t neon_diff6 = vsubq_f16(neon_base6, neon_query);
        float16x8_t neon_diff7 = vsubq_f16(neon_base7, neon_query);
        float16x8_t neon_diff8 = vsubq_f16(neon_base8, neon_query);

        neon_res1 = vfmlalq_low_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_low_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_low_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_low_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_low_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_low_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_low_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_low_f16(neon_res8, neon_diff8, neon_diff8);

        neon_res1 = vfmlalq_high_f16(neon_res1, neon_diff1, neon_diff1);
        neon_res2 = vfmlalq_high_f16(neon_res2, neon_diff2, neon_diff2);
        neon_res3 = vfmlalq_high_f16(neon_res3, neon_diff3, neon_diff3);
        neon_res4 = vfmlalq_high_f16(neon_res4, neon_diff4, neon_diff4);
        neon_res5 = vfmlalq_high_f16(neon_res5, neon_diff5, neon_diff5);
        neon_res6 = vfmlalq_high_f16(neon_res6, neon_diff6, neon_diff6);
        neon_res7 = vfmlalq_high_f16(neon_res7, neon_diff7, neon_diff7);
        neon_res8 = vfmlalq_high_f16(neon_res8, neon_diff8, neon_diff8);
        dis[0] = vaddvq_f32(neon_res1);
        dis[1] = vaddvq_f32(neon_res2);
        dis[2] = vaddvq_f32(neon_res3);
        dis[3] = vaddvq_f32(neon_res4);
        dis[4] = vaddvq_f32(neon_res5);
        dis[5] = vaddvq_f32(neon_res6);
        dis[6] = vaddvq_f32(neon_res7);
        dis[7] = vaddvq_f32(neon_res8);
        i = single_round;
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float16_t q4 = x[i] - *(y + 4 * d + i);
        float16_t q5 = x[i] - *(y + 5 * d + i);
        float16_t q6 = x[i] - *(y + 6 * d + i);
        float16_t q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
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
static void krl_L2sqr_batch16_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

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
        for (i = 0; i < d - multi_round; i += multi_round) {
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
            for (size_t j = 0; j < multi_round; j += single_round) {
                const float16x8_t neon_query = vld1q_f16(x + i + j);
                float16x8_t neon_base1 = vld1q_f16(y + i + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + i + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i + j);
                float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i + j);
                float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i + j);
                float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i + j);
                float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i + j);

                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_base5 = vsubq_f16(neon_base5, neon_query);
                neon_base6 = vsubq_f16(neon_base6, neon_query);
                neon_base7 = vsubq_f16(neon_base7, neon_query);
                neon_base8 = vsubq_f16(neon_base8, neon_query);

                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

                neon_base1 = vld1q_f16(y + 8 * d + i + j);
                neon_base2 = vld1q_f16(y + 9 * d + i + j);
                neon_base3 = vld1q_f16(y + 10 * d + i + j);
                neon_base4 = vld1q_f16(y + 11 * d + i + j);
                neon_base5 = vld1q_f16(y + 12 * d + i + j);
                neon_base6 = vld1q_f16(y + 13 * d + i + j);
                neon_base7 = vld1q_f16(y + 14 * d + i + j);
                neon_base8 = vld1q_f16(y + 15 * d + i + j);

                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_base5 = vsubq_f16(neon_base5, neon_query);
                neon_base6 = vsubq_f16(neon_base6, neon_query);
                neon_base7 = vsubq_f16(neon_base7, neon_query);
                neon_base8 = vsubq_f16(neon_base8, neon_query);

                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
            }
        }
        for (; i + single_round <= d; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
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
    } else {
        for (i = 0; i + single_round <= d; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            float16x8_t neon_base5 = vld1q_f16(y + 4 * d + i);
            float16x8_t neon_base6 = vld1q_f16(y + 5 * d + i);
            float16x8_t neon_base7 = vld1q_f16(y + 6 * d + i);
            float16x8_t neon_base8 = vld1q_f16(y + 7 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base8, neon_base8);

            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base5, neon_base5);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base6, neon_base6);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base7, neon_base7);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base8, neon_base8);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base5 = vld1q_f16(y + 12 * d + i);
            neon_base6 = vld1q_f16(y + 13 * d + i);
            neon_base7 = vld1q_f16(y + 14 * d + i);
            neon_base8 = vld1q_f16(y + 15 * d + i);

            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_base5 = vsubq_f16(neon_base5, neon_query);
            neon_base6 = vsubq_f16(neon_base6, neon_query);
            neon_base7 = vsubq_f16(neon_base7, neon_query);
            neon_base8 = vsubq_f16(neon_base8, neon_query);

            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base8, neon_base8);

            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base5, neon_base5);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base6, neon_base6);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base7, neon_base7);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base8, neon_base8);
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
    }
    if (i < d) {
        float16_t q0 = x[i] - *(y + i);
        float16_t q1 = x[i] - *(y + d + i);
        float16_t q2 = x[i] - *(y + 2 * d + i);
        float16_t q3 = x[i] - *(y + 3 * d + i);
        float16_t q4 = x[i] - *(y + 4 * d + i);
        float16_t q5 = x[i] - *(y + 5 * d + i);
        float16_t q6 = x[i] - *(y + 6 * d + i);
        float16_t q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y + 8 * d + i);
            q1 = x[i] - *(y + 9 * d + i);
            q2 = x[i] - *(y + 10 * d + i);
            q3 = x[i] - *(y + 11 * d + i);
            q4 = x[i] - *(y + 12 * d + i);
            q5 = x[i] - *(y + 13 * d + i);
            q6 = x[i] - *(y + 14 * d + i);
            q7 = x[i] - *(y + 15 * d + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
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

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch24_f16f32(const float16_t *x, const float16_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 8; /* 128 / 16 */
    constexpr size_t multi_round = 32; /* 4 * single_round */

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
                const float16x8_t neon_query = vld1q_f16(x + j);
                float16x8_t neon_base1 = vld1q_f16(y + j);
                float16x8_t neon_base2 = vld1q_f16(y + d + j);
                float16x8_t neon_base3 = vld1q_f16(y + 2 * d + j);
                float16x8_t neon_base4 = vld1q_f16(y + 3 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
                neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
                neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
                neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
                neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 4 * d + j);
                neon_base2 = vld1q_f16(y + 5 * d + j);
                neon_base3 = vld1q_f16(y + 6 * d + j);
                neon_base4 = vld1q_f16(y + 7 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
                neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
                neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
                neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
                neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 8 * d + j);
                neon_base2 = vld1q_f16(y + 9 * d + j);
                neon_base3 = vld1q_f16(y + 10 * d + j);
                neon_base4 = vld1q_f16(y + 11 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
                neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
                neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
                neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
                neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 12 * d + j);
                neon_base2 = vld1q_f16(y + 13 * d + j);
                neon_base3 = vld1q_f16(y + 14 * d + j);
                neon_base4 = vld1q_f16(y + 15 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
                neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
                neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
                neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
                neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 16 * d + j);
                neon_base2 = vld1q_f16(y + 17 * d + j);
                neon_base3 = vld1q_f16(y + 18 * d + j);
                neon_base4 = vld1q_f16(y + 19 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
                neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
                neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
                neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
                neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

                neon_base1 = vld1q_f16(y + 20 * d + j);
                neon_base2 = vld1q_f16(y + 21 * d + j);
                neon_base3 = vld1q_f16(y + 22 * d + j);
                neon_base4 = vld1q_f16(y + 23 * d + j);
                neon_base1 = vsubq_f16(neon_base1, neon_query);
                neon_base2 = vsubq_f16(neon_base2, neon_query);
                neon_base3 = vsubq_f16(neon_base3, neon_query);
                neon_base4 = vsubq_f16(neon_base4, neon_query);
                neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
                neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
                neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
                neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
                neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
            }
        }
        for (; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
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
    } else if (d >= single_round) {
        for (i = 0; i <= d - single_round; i += single_round) {
            const float16x8_t neon_query = vld1q_f16(x + i);
            float16x8_t neon_base1 = vld1q_f16(y + i);
            float16x8_t neon_base2 = vld1q_f16(y + d + i);
            float16x8_t neon_base3 = vld1q_f16(y + 2 * d + i);
            float16x8_t neon_base4 = vld1q_f16(y + 3 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res1 = vfmlalq_low_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_low_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_low_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_low_f16(neon_res4, neon_base4, neon_base4);
            neon_res1 = vfmlalq_high_f16(neon_res1, neon_base1, neon_base1);
            neon_res2 = vfmlalq_high_f16(neon_res2, neon_base2, neon_base2);
            neon_res3 = vfmlalq_high_f16(neon_res3, neon_base3, neon_base3);
            neon_res4 = vfmlalq_high_f16(neon_res4, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 4 * d + i);
            neon_base2 = vld1q_f16(y + 5 * d + i);
            neon_base3 = vld1q_f16(y + 6 * d + i);
            neon_base4 = vld1q_f16(y + 7 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res5 = vfmlalq_low_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_low_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_low_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_low_f16(neon_res8, neon_base4, neon_base4);
            neon_res5 = vfmlalq_high_f16(neon_res5, neon_base1, neon_base1);
            neon_res6 = vfmlalq_high_f16(neon_res6, neon_base2, neon_base2);
            neon_res7 = vfmlalq_high_f16(neon_res7, neon_base3, neon_base3);
            neon_res8 = vfmlalq_high_f16(neon_res8, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 8 * d + i);
            neon_base2 = vld1q_f16(y + 9 * d + i);
            neon_base3 = vld1q_f16(y + 10 * d + i);
            neon_base4 = vld1q_f16(y + 11 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res9 = vfmlalq_low_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_low_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_low_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_low_f16(neon_res12, neon_base4, neon_base4);
            neon_res9 = vfmlalq_high_f16(neon_res9, neon_base1, neon_base1);
            neon_res10 = vfmlalq_high_f16(neon_res10, neon_base2, neon_base2);
            neon_res11 = vfmlalq_high_f16(neon_res11, neon_base3, neon_base3);
            neon_res12 = vfmlalq_high_f16(neon_res12, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 12 * d + i);
            neon_base2 = vld1q_f16(y + 13 * d + i);
            neon_base3 = vld1q_f16(y + 14 * d + i);
            neon_base4 = vld1q_f16(y + 15 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res13 = vfmlalq_low_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_low_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_low_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_low_f16(neon_res16, neon_base4, neon_base4);
            neon_res13 = vfmlalq_high_f16(neon_res13, neon_base1, neon_base1);
            neon_res14 = vfmlalq_high_f16(neon_res14, neon_base2, neon_base2);
            neon_res15 = vfmlalq_high_f16(neon_res15, neon_base3, neon_base3);
            neon_res16 = vfmlalq_high_f16(neon_res16, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 16 * d + i);
            neon_base2 = vld1q_f16(y + 17 * d + i);
            neon_base3 = vld1q_f16(y + 18 * d + i);
            neon_base4 = vld1q_f16(y + 19 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res17 = vfmlalq_low_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_low_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_low_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_low_f16(neon_res20, neon_base4, neon_base4);
            neon_res17 = vfmlalq_high_f16(neon_res17, neon_base1, neon_base1);
            neon_res18 = vfmlalq_high_f16(neon_res18, neon_base2, neon_base2);
            neon_res19 = vfmlalq_high_f16(neon_res19, neon_base3, neon_base3);
            neon_res20 = vfmlalq_high_f16(neon_res20, neon_base4, neon_base4);

            neon_base1 = vld1q_f16(y + 20 * d + i);
            neon_base2 = vld1q_f16(y + 21 * d + i);
            neon_base3 = vld1q_f16(y + 22 * d + i);
            neon_base4 = vld1q_f16(y + 23 * d + i);
            neon_base1 = vsubq_f16(neon_base1, neon_query);
            neon_base2 = vsubq_f16(neon_base2, neon_query);
            neon_base3 = vsubq_f16(neon_base3, neon_query);
            neon_base4 = vsubq_f16(neon_base4, neon_query);
            neon_res21 = vfmlalq_low_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_low_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_low_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_low_f16(neon_res24, neon_base4, neon_base4);
            neon_res21 = vfmlalq_high_f16(neon_res21, neon_base1, neon_base1);
            neon_res22 = vfmlalq_high_f16(neon_res22, neon_base2, neon_base2);
            neon_res23 = vfmlalq_high_f16(neon_res23, neon_base3, neon_base3);
            neon_res24 = vfmlalq_high_f16(neon_res24, neon_base4, neon_base4);
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
        memset(dis, 0, sizeof(float) * 24);
        i = 0;
    }
    if (i < d) {
        float q0 = x[i] - *(y + i);
        float q1 = x[i] - *(y + d + i);
        float q2 = x[i] - *(y + 2 * d + i);
        float q3 = x[i] - *(y + 3 * d + i);
        float q4 = x[i] - *(y + 4 * d + i);
        float q5 = x[i] - *(y + 5 * d + i);
        float q6 = x[i] - *(y + 6 * d + i);
        float q7 = x[i] - *(y + 7 * d + i);
        float d0 = q0 * q0;
        float d1 = q1 * q1;
        float d2 = q2 * q2;
        float d3 = q3 * q3;
        float d4 = q4 * q4;
        float d5 = q5 * q5;
        float d6 = q6 * q6;
        float d7 = q7 * q7;
        q0 = x[i] - *(y + 8 * d + i);
        q1 = x[i] - *(y + 9 * d + i);
        q2 = x[i] - *(y + 10 * d + i);
        q3 = x[i] - *(y + 11 * d + i);
        q4 = x[i] - *(y + 12 * d + i);
        q5 = x[i] - *(y + 13 * d + i);
        q6 = x[i] - *(y + 14 * d + i);
        q7 = x[i] - *(y + 15 * d + i);
        float d8 = q0 * q0;
        float d9 = q1 * q1;
        float d10 = q2 * q2;
        float d11 = q3 * q3;
        float d12 = q4 * q4;
        float d13 = q5 * q5;
        float d14 = q6 * q6;
        float d15 = q7 * q7;
        q0 = x[i] - *(y + 16 * d + i);
        q1 = x[i] - *(y + 17 * d + i);
        q2 = x[i] - *(y + 18 * d + i);
        q3 = x[i] - *(y + 19 * d + i);
        q4 = x[i] - *(y + 20 * d + i);
        q5 = x[i] - *(y + 21 * d + i);
        q6 = x[i] - *(y + 22 * d + i);
        q7 = x[i] - *(y + 23 * d + i);
        float d16 = q0 * q0;
        float d17 = q1 * q1;
        float d18 = q2 * q2;
        float d19 = q3 * q3;
        float d20 = q4 * q4;
        float d21 = q5 * q5;
        float d22 = q6 * q6;
        float d23 = q7 * q7;
        for (i++; i < d; ++i) {
            q0 = x[i] - *(y + i);
            q1 = x[i] - *(y + d + i);
            q2 = x[i] - *(y + 2 * d + i);
            q3 = x[i] - *(y + 3 * d + i);
            q4 = x[i] - *(y + 4 * d + i);
            q5 = x[i] - *(y + 5 * d + i);
            q6 = x[i] - *(y + 6 * d + i);
            q7 = x[i] - *(y + 7 * d + i);
            d0 += q0 * q0;
            d1 += q1 * q1;
            d2 += q2 * q2;
            d3 += q3 * q3;
            d4 += q4 * q4;
            d5 += q5 * q5;
            d6 += q6 * q6;
            d7 += q7 * q7;
            q0 = x[i] - *(y + 8 * d + i);
            q1 = x[i] - *(y + 9 * d + i);
            q2 = x[i] - *(y + 10 * d + i);
            q3 = x[i] - *(y + 11 * d + i);
            q4 = x[i] - *(y + 12 * d + i);
            q5 = x[i] - *(y + 13 * d + i);
            q6 = x[i] - *(y + 14 * d + i);
            q7 = x[i] - *(y + 15 * d + i);
            d8 += q0 * q0;
            d9 += q1 * q1;
            d10 += q2 * q2;
            d11 += q3 * q3;
            d12 += q4 * q4;
            d13 += q5 * q5;
            d14 += q6 * q6;
            d15 += q7 * q7;
            q0 = x[i] - *(y + 16 * d + i);
            q1 = x[i] - *(y + 17 * d + i);
            q2 = x[i] - *(y + 18 * d + i);
            q3 = x[i] - *(y + 19 * d + i);
            q4 = x[i] - *(y + 20 * d + i);
            q5 = x[i] - *(y + 21 * d + i);
            q6 = x[i] - *(y + 22 * d + i);
            q7 = x[i] - *(y + 23 * d + i);
            d16 += q0 * q0;
            d17 += q1 * q1;
            d18 += q2 * q2;
            d19 += q3 * q3;
            d20 += q4 * q4;
            d21 += q5 * q5;
            d22 += q6 * q6;
            d23 += q7 * q7;
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

int krl_L2sqr_ny_f16f32(float *dis, const uint16_t *x, const uint16_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    for (; i + 24 <= ny; i += 24) {
        prefetch_L1(x);
        prefetch_Lx(y + i * d);
        krl_L2sqr_batch24_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
    }
    if (i + 16 <= ny) {
        prefetch_L1(x);
        prefetch_Lx(y + i * d);
        krl_L2sqr_batch16_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 16;
    } else if (i + 8 <= ny) {
        krl_L2sqr_batch8_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_L2sqr_batch4_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_L2sqr_batch2_f16f32((const float16_t *)x, (const float16_t *)y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        krl_L2sqr_f16f32(x, y + i * d, d, &dis[i], 1);
    }
    return SUCCESS;
}