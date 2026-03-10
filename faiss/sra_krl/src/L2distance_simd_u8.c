#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include <stdio.h>

KRL_IMPRECISE_FUNCTION_BEGIN
int krl_L2sqr_u8u32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, uint32_t *dis, size_t dis_size)
{
    size_t i;
    uint32_t res;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 64;

    uint32x4_t res1 = vdupq_n_u32(0);
    uint32x4_t res2 = vdupq_n_u32(0);
    uint32x4_t res3 = vdupq_n_u32(0);
    uint32x4_t res4 = vdupq_n_u32(0);
    for (i = 0; i + double_round <= d; i += double_round) {
        const uint8x16_t x8_0 = vld1q_u8(x + i);
        const uint8x16_t x8_1 = vld1q_u8(x + i + 16);
        const uint8x16_t x8_2 = vld1q_u8(x + i + 32);
        const uint8x16_t x8_3 = vld1q_u8(x + i + 48);

        const uint8x16_t y8_0 = vld1q_u8(y + i);
        const uint8x16_t y8_1 = vld1q_u8(y + i + 16);
        const uint8x16_t y8_2 = vld1q_u8(y + i + 32);
        const uint8x16_t y8_3 = vld1q_u8(y + i + 48);

        const uint8x16_t d8_0 = vabdq_u8(x8_0, y8_0);
        const uint8x16_t d8_1 = vabdq_u8(x8_1, y8_1);
        const uint8x16_t d8_2 = vabdq_u8(x8_2, y8_2);
        const uint8x16_t d8_3 = vabdq_u8(x8_3, y8_3);

        res1 = vdotq_u32(res1, d8_0, d8_0);
        res2 = vdotq_u32(res2, d8_1, d8_1);
        res3 = vdotq_u32(res3, d8_2, d8_2);
        res4 = vdotq_u32(res4, d8_3, d8_3);
    }
    for (; i + single_round <= d; i += single_round) {
        const uint8x16_t x8_0 = vld1q_u8(x + i);
        const uint8x16_t y8_0 = vld1q_u8(y + i);

        const uint8x16_t d8_0 = vabdq_u8(x8_0, y8_0);
        res1 = vdotq_u32(res1, d8_0, d8_0);
    }
    res1 = vaddq_u32(res1, res2);
    res3 = vaddq_u32(res3, res4);
    res1 = vaddq_u32(res1, res3);
    res = vaddvq_u32(res1);
    for (; i < d; i++) {
        const int32_t tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    *dis = res;
    return SUCCESS;
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch2_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;

    if (likely(d >= single_round)) {
        uint32x4_t res00 = vdupq_n_u32(0);
        uint32x4_t res01 = vdupq_n_u32(0);
        uint32x4_t res10 = vdupq_n_u32(0);
        uint32x4_t res11 = vdupq_n_u32(0);
        for (i = 0; i + double_round <= d; i += double_round) {
            const uint8x16_t x0 = vld1q_u8(x + i);
            const uint8x16_t x1 = vld1q_u8(x + i + 16);
            const uint8x16_t y00 = vld1q_u8(y + i);
            const uint8x16_t y01 = vld1q_u8(y + i + 16);
            const uint8x16_t y10 = vld1q_u8(y + i + d);
            const uint8x16_t y11 = vld1q_u8(y + i + d + 16);

            const uint8x16_t d00 = vabdq_u8(x0, y00);
            res00 = vdotq_u32(res00, d00, d00);
            const uint8x16_t d01 = vabdq_u8(x1, y01);
            res01 = vdotq_u32(res01, d01, d01);
            const uint8x16_t d10 = vabdq_u8(x0, y10);
            res10 = vdotq_u32(res10, d10, d10);
            const uint8x16_t d11 = vabdq_u8(x1, y11);
            res11 = vdotq_u32(res11, d11, d11);
        }
        for (; i <= d - single_round; i += single_round) {
            const uint8x16_t x0 = vld1q_u8(x + i);
            const uint8x16_t y00 = vld1q_u8(y + i);
            const uint8x16_t y10 = vld1q_u8(y + i + d);

            const uint8x16_t d00 = vabdq_u8(x0, y00);
            res00 = vdotq_u32(res00, d00, d00);
            const uint8x16_t d10 = vabdq_u8(x0, y10);
            res10 = vdotq_u32(res10, d10, d10);
        }
        res00 = vaddq_u32(res00, res01);
        res10 = vaddq_u32(res10, res11);
        dis[0] = (float)vaddvq_u32(res00);
        dis[1] = (float)vaddvq_u32(res10);
    } else {
        dis[0] = 0;
        dis[1] = 0;
        i = 0;
    }

    for (; i < d; i++) {
        const float tmp0 = x[i] - *(y + i);
        const float tmp1 = x[i] - *(y + d + i);
        dis[0] += tmp0 * tmp0;
        dis[1] += tmp1 * tmp1;
    }
}
KRL_IMPRECISE_FUNCTION_END

KRL_IMPRECISE_FUNCTION_BEGIN
static void krl_L2sqr_batch4_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint8x16_t neon_query;
    uint8x16_t neon_base[4];
    uint8x16_t neon_diff[4];
    uint32x4_t neon_res[4];
    if (likely(d >= single_round)) {
        neon_query = vld1q_u8(x);
        neon_base[0] = vld1q_u8(y);
        neon_base[1] = vld1q_u8(y + d);
        neon_base[2] = vld1q_u8(y + 2 * d);
        neon_base[3] = vld1q_u8(y + 3 * d);

        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);

        neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
        neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
        neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
        neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
        if (d < double_round) {
            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            i = single_round;
        } else {
            neon_query = vld1q_u8(x + single_round);
            neon_base[0] = vld1q_u8(y + single_round);
            neon_base[1] = vld1q_u8(y + d + single_round);
            neon_base[2] = vld1q_u8(y + 2 * d + single_round);
            neon_base[3] = vld1q_u8(y + 3 * d + single_round);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            for (i = double_round; i <= d - single_round; i += single_round) {
                neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
                neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
                neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
                neon_diff[3] = vabdq_u8(neon_base[3], neon_query);

                neon_query = vld1q_u8(x + i);
                neon_base[0] = vld1q_u8(y + i);
                neon_base[1] = vld1q_u8(y + d + i);
                neon_base[2] = vld1q_u8(y + 2 * d + i);
                neon_base[3] = vld1q_u8(y + 3 * d + i);

                neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
                neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
                neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
                neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            }
            neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
            neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
            neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
            neon_diff[3] = vabdq_u8(neon_base[3], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
        }
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[1]);
        neon_res[2] = vpaddq_u32(neon_res[2], neon_res[3]);
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[2]);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res[0]));
    } else {
        for (int i = 0; i < 4; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
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
static void krl_L2sqr_batch8_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16;
    constexpr size_t double_round = 32;
    uint8x16_t neon_query;
    uint8x16_t neon_base[8];
    uint8x16_t neon_diff[8];
    uint32x4_t neon_res[8];
    if (likely(d >= single_round)) {
        neon_query = vld1q_u8(x);
        neon_base[0] = vld1q_u8(y);
        neon_base[1] = vld1q_u8(y + d);
        neon_base[2] = vld1q_u8(y + 2 * d);
        neon_base[3] = vld1q_u8(y + 3 * d);
        neon_base[4] = vld1q_u8(y + 4 * d);
        neon_base[5] = vld1q_u8(y + 5 * d);
        neon_base[6] = vld1q_u8(y + 6 * d);
        neon_base[7] = vld1q_u8(y + 7 * d);

        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);
        neon_res[4] = vdupq_n_u32(0);
        neon_res[5] = vdupq_n_u32(0);
        neon_res[6] = vdupq_n_u32(0);
        neon_res[7] = vdupq_n_u32(0);

        neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
        neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
        neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
        neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
        neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
        neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
        neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
        neon_diff[7] = vabdq_u8(neon_base[7], neon_query);
        if (d < double_round) {
            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            i = single_round;
        } else {
            neon_query = vld1q_u8(x + single_round);
            neon_base[0] = vld1q_u8(y + single_round);
            neon_base[1] = vld1q_u8(y + d + single_round);
            neon_base[2] = vld1q_u8(y + 2 * d + single_round);
            neon_base[3] = vld1q_u8(y + 3 * d + single_round);
            neon_base[4] = vld1q_u8(y + 4 * d + single_round);
            neon_base[5] = vld1q_u8(y + 5 * d + single_round);
            neon_base[6] = vld1q_u8(y + 6 * d + single_round);
            neon_base[7] = vld1q_u8(y + 7 * d + single_round);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            for (i = double_round; i <= d - single_round; i += single_round) {
                neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
                neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
                neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
                neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
                neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
                neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
                neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
                neon_diff[7] = vabdq_u8(neon_base[7], neon_query);

                neon_query = vld1q_u8(x + i);
                neon_base[0] = vld1q_u8(y + i);
                neon_base[1] = vld1q_u8(y + d + i);
                neon_base[2] = vld1q_u8(y + 2 * d + i);
                neon_base[3] = vld1q_u8(y + 3 * d + i);
                neon_base[4] = vld1q_u8(y + 4 * d + i);
                neon_base[5] = vld1q_u8(y + 5 * d + i);
                neon_base[6] = vld1q_u8(y + 6 * d + i);
                neon_base[7] = vld1q_u8(y + 7 * d + i);

                neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
                neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
                neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
                neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
                neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
                neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
                neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
                neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
            }
            neon_diff[0] = vabdq_u8(neon_base[0], neon_query);
            neon_diff[1] = vabdq_u8(neon_base[1], neon_query);
            neon_diff[2] = vabdq_u8(neon_base[2], neon_query);
            neon_diff[3] = vabdq_u8(neon_base[3], neon_query);
            neon_diff[4] = vabdq_u8(neon_base[4], neon_query);
            neon_diff[5] = vabdq_u8(neon_base[5], neon_query);
            neon_diff[6] = vabdq_u8(neon_base[6], neon_query);
            neon_diff[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_diff[0], neon_diff[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_diff[1], neon_diff[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_diff[2], neon_diff[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_diff[3], neon_diff[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_diff[4], neon_diff[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_diff[5], neon_diff[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_diff[6], neon_diff[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_diff[7], neon_diff[7]);
        }
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[1]);
        neon_res[2] = vpaddq_u32(neon_res[2], neon_res[3]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[5]);
        neon_res[6] = vpaddq_u32(neon_res[6], neon_res[7]);
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[2]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[6]);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res[0]));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res[4]));
    } else {
        for (int i = 0; i < 8; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        int32_t q4 = x[i] - *(y + 4 * d + i);
        int32_t q5 = x[i] - *(y + 5 * d + i);
        int32_t q6 = x[i] - *(y + 6 * d + i);
        int32_t q7 = x[i] - *(y + 7 * d + i);
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
static void krl_L2sqr_batch16_u8f32(const uint8_t *x, const uint8_t *__restrict y, const size_t d, float *dis)
{
    size_t i;
    constexpr size_t single_round = 16; /* 128 / 8 */
    uint8x16_t neon_query;
    uint8x16_t neon_base[8];
    uint32x4_t neon_res[16];
    if (likely(d >= single_round)) {
        neon_res[0] = vdupq_n_u32(0);
        neon_res[1] = vdupq_n_u32(0);
        neon_res[2] = vdupq_n_u32(0);
        neon_res[3] = vdupq_n_u32(0);
        neon_res[4] = vdupq_n_u32(0);
        neon_res[5] = vdupq_n_u32(0);
        neon_res[6] = vdupq_n_u32(0);
        neon_res[7] = vdupq_n_u32(0);
        neon_res[8] = vdupq_n_u32(0);
        neon_res[9] = vdupq_n_u32(0);
        neon_res[10] = vdupq_n_u32(0);
        neon_res[11] = vdupq_n_u32(0);
        neon_res[12] = vdupq_n_u32(0);
        neon_res[13] = vdupq_n_u32(0);
        neon_res[14] = vdupq_n_u32(0);
        neon_res[15] = vdupq_n_u32(0);
        for (i = 0; i <= d - single_round; i += single_round) {
            neon_query = vld1q_u8(x + i);
            neon_base[0] = vld1q_u8(y + i);
            neon_base[1] = vld1q_u8(y + d + i);
            neon_base[2] = vld1q_u8(y + 2 * d + i);
            neon_base[3] = vld1q_u8(y + 3 * d + i);
            neon_base[4] = vld1q_u8(y + 4 * d + i);
            neon_base[5] = vld1q_u8(y + 5 * d + i);
            neon_base[6] = vld1q_u8(y + 6 * d + i);
            neon_base[7] = vld1q_u8(y + 7 * d + i);

            neon_base[0] = vabdq_u8(neon_base[0], neon_query);
            neon_base[1] = vabdq_u8(neon_base[1], neon_query);
            neon_base[2] = vabdq_u8(neon_base[2], neon_query);
            neon_base[3] = vabdq_u8(neon_base[3], neon_query);
            neon_base[4] = vabdq_u8(neon_base[4], neon_query);
            neon_base[5] = vabdq_u8(neon_base[5], neon_query);
            neon_base[6] = vabdq_u8(neon_base[6], neon_query);
            neon_base[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[0] = vdotq_u32(neon_res[0], neon_base[0], neon_base[0]);
            neon_res[1] = vdotq_u32(neon_res[1], neon_base[1], neon_base[1]);
            neon_res[2] = vdotq_u32(neon_res[2], neon_base[2], neon_base[2]);
            neon_res[3] = vdotq_u32(neon_res[3], neon_base[3], neon_base[3]);
            neon_res[4] = vdotq_u32(neon_res[4], neon_base[4], neon_base[4]);
            neon_res[5] = vdotq_u32(neon_res[5], neon_base[5], neon_base[5]);
            neon_res[6] = vdotq_u32(neon_res[6], neon_base[6], neon_base[6]);
            neon_res[7] = vdotq_u32(neon_res[7], neon_base[7], neon_base[7]);

            neon_base[0] = vld1q_u8(y + 8 * d + i);
            neon_base[1] = vld1q_u8(y + 9 * d + i);
            neon_base[2] = vld1q_u8(y + 10 * d + i);
            neon_base[3] = vld1q_u8(y + 11 * d + i);
            neon_base[4] = vld1q_u8(y + 12 * d + i);
            neon_base[5] = vld1q_u8(y + 13 * d + i);
            neon_base[6] = vld1q_u8(y + 14 * d + i);
            neon_base[7] = vld1q_u8(y + 15 * d + i);

            neon_base[0] = vabdq_u8(neon_base[0], neon_query);
            neon_base[1] = vabdq_u8(neon_base[1], neon_query);
            neon_base[2] = vabdq_u8(neon_base[2], neon_query);
            neon_base[3] = vabdq_u8(neon_base[3], neon_query);
            neon_base[4] = vabdq_u8(neon_base[4], neon_query);
            neon_base[5] = vabdq_u8(neon_base[5], neon_query);
            neon_base[6] = vabdq_u8(neon_base[6], neon_query);
            neon_base[7] = vabdq_u8(neon_base[7], neon_query);

            neon_res[8] = vdotq_u32(neon_res[8], neon_base[0], neon_base[0]);
            neon_res[9] = vdotq_u32(neon_res[9], neon_base[1], neon_base[1]);
            neon_res[10] = vdotq_u32(neon_res[10], neon_base[2], neon_base[2]);
            neon_res[11] = vdotq_u32(neon_res[11], neon_base[3], neon_base[3]);
            neon_res[12] = vdotq_u32(neon_res[12], neon_base[4], neon_base[4]);
            neon_res[13] = vdotq_u32(neon_res[13], neon_base[5], neon_base[5]);
            neon_res[14] = vdotq_u32(neon_res[14], neon_base[6], neon_base[6]);
            neon_res[15] = vdotq_u32(neon_res[15], neon_base[7], neon_base[7]);
        }
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[1]);
        neon_res[2] = vpaddq_u32(neon_res[2], neon_res[3]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[5]);
        neon_res[6] = vpaddq_u32(neon_res[6], neon_res[7]);
        neon_res[8] = vpaddq_u32(neon_res[8], neon_res[9]);
        neon_res[10] = vpaddq_u32(neon_res[10], neon_res[11]);
        neon_res[12] = vpaddq_u32(neon_res[12], neon_res[13]);
        neon_res[14] = vpaddq_u32(neon_res[14], neon_res[15]);
        neon_res[0] = vpaddq_u32(neon_res[0], neon_res[2]);
        neon_res[4] = vpaddq_u32(neon_res[4], neon_res[6]);
        neon_res[8] = vpaddq_u32(neon_res[8], neon_res[10]);
        neon_res[12] = vpaddq_u32(neon_res[12], neon_res[14]);
        vst1q_f32(dis, vcvtq_f32_u32(neon_res[0]));
        vst1q_f32(dis + 4, vcvtq_f32_u32(neon_res[4]));
        vst1q_f32(dis + 8, vcvtq_f32_u32(neon_res[8]));
        vst1q_f32(dis + 12, vcvtq_f32_u32(neon_res[12]));
    } else {
        for (int i = 0; i < 16; i++) {
            dis[i] = 0.0f;
        }
        i = 0;
    }
    if (i < d) {
        int32_t q0 = x[i] - *(y + i);
        int32_t q1 = x[i] - *(y + d + i);
        int32_t q2 = x[i] - *(y + 2 * d + i);
        int32_t q3 = x[i] - *(y + 3 * d + i);
        int32_t q4 = x[i] - *(y + 4 * d + i);
        int32_t q5 = x[i] - *(y + 5 * d + i);
        int32_t q6 = x[i] - *(y + 6 * d + i);
        int32_t q7 = x[i] - *(y + 7 * d + i);
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

int krl_L2sqr_ny_u8f32(float *dis, const uint8_t *x, const uint8_t *y, size_t ny, size_t d, size_t dis_size)
{
    size_t i = 0;

    for (; i + 16 <= ny; i += 16) {
        krl_L2sqr_batch16_u8f32(x, y + i * d, d, dis + i);
    }
    if (ny & 8) {
        krl_L2sqr_batch8_u8f32(x, y + i * d, d, dis + i);
        i += 8;
    }
    if (ny & 4) {
        krl_L2sqr_batch4_u8f32(x, y + i * d, d, dis + i);
        i += 4;
    }
    if (ny & 2) {
        krl_L2sqr_batch2_u8f32(x, y + i * d, d, dis + i);
        i += 2;
    }
    if (ny & 1) {
        uint32_t tmp;
        krl_L2sqr_u8u32(x, y + i * d, d, &tmp, 1);
        dis[i] = (float)tmp;
    }
    return SUCCESS;
}