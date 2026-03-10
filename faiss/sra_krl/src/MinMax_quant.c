#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"

void quant_f16(const float *src, idx_t n, float16_t *out)
{
    idx_t l = 0;
    const idx_t single_loop = 4;
    const idx_t multi_loop = 16;
    for (; l + multi_loop <= n; l += multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        const float16x4_t neon_c1 = vcvt_f16_f32(neon_a1);
        const float16x4_t neon_c2 = vcvt_f16_f32(neon_a2);
        const float16x4_t neon_c3 = vcvt_f16_f32(neon_a3);
        const float16x4_t neon_c4 = vcvt_f16_f32(neon_a4);
        vst1_f16(out + l, neon_c1);
        vst1_f16(out + l + 4, neon_c2);
        vst1_f16(out + l + 8, neon_c3);
        vst1_f16(out + l + 12, neon_c4);
    }
    for (; l + single_loop <= n; l += single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        const float16x4_t neon_c1 = vcvt_f16_f32(neon_a1);
        vst1_f16(out + l, neon_c1);
    }
    for (; l < n; ++l) {
        out[l] = (float16_t)(src[l]);
    }
}

void quant_u8(const float *src, idx_t n, uint8_t *out)
{
    idx_t l;
    const idx_t multi_loop = 32;
    const idx_t double_loop = 16;
    const idx_t single_loop = 8;
    if (n >= multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src);
        float32x4_t neon_a2 = vld1q_f32(src + 4);
        float32x4_t neon_a3 = vld1q_f32(src + 8);
        float32x4_t neon_a4 = vld1q_f32(src + 12);
        float32x4_t neon_a5 = vld1q_f32(src + 16);
        float32x4_t neon_a6 = vld1q_f32(src + 20);
        float32x4_t neon_a7 = vld1q_f32(src + 24);
        float32x4_t neon_a8 = vld1q_f32(src + 28);
        for (l = multi_loop; l + multi_loop <= n; l += multi_loop) {
            const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
            const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);
            const uint32x4_t neon_b3 = vcvtaq_u32_f32(neon_a3);
            const uint32x4_t neon_b4 = vcvtaq_u32_f32(neon_a4);
            const uint32x4_t neon_b5 = vcvtaq_u32_f32(neon_a5);
            const uint32x4_t neon_b6 = vcvtaq_u32_f32(neon_a6);
            const uint32x4_t neon_b7 = vcvtaq_u32_f32(neon_a7);
            const uint32x4_t neon_b8 = vcvtaq_u32_f32(neon_a8);

            neon_a1 = vld1q_f32(src + l);
            neon_a2 = vld1q_f32(src + l + 4);
            neon_a3 = vld1q_f32(src + l + 8);
            neon_a4 = vld1q_f32(src + l + 12);
            neon_a5 = vld1q_f32(src + l + 16);
            neon_a6 = vld1q_f32(src + l + 20);
            neon_a7 = vld1q_f32(src + l + 24);
            neon_a8 = vld1q_f32(src + l + 28);

            const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
            const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));
            const uint8x16_t neon_c3 = vpaddq_u8(vreinterpretq_u8_u32(neon_b5), vreinterpretq_u8_u32(neon_b6));
            const uint8x16_t neon_c4 = vpaddq_u8(vreinterpretq_u8_u32(neon_b7), vreinterpretq_u8_u32(neon_b8));

            const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);
            const uint8x16_t neon_d2 = vpaddq_u8(neon_c3, neon_c4);

            vst1q_u8(out + l - multi_loop, neon_d1);
            vst1q_u8(out + l - multi_loop + 16, neon_d2);
        }
        const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);
        const uint32x4_t neon_b3 = vcvtaq_u32_f32(neon_a3);
        const uint32x4_t neon_b4 = vcvtaq_u32_f32(neon_a4);
        const uint32x4_t neon_b5 = vcvtaq_u32_f32(neon_a5);
        const uint32x4_t neon_b6 = vcvtaq_u32_f32(neon_a6);
        const uint32x4_t neon_b7 = vcvtaq_u32_f32(neon_a7);
        const uint32x4_t neon_b8 = vcvtaq_u32_f32(neon_a8);

        const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
        const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));
        const uint8x16_t neon_c3 = vpaddq_u8(vreinterpretq_u8_u32(neon_b5), vreinterpretq_u8_u32(neon_b6));
        const uint8x16_t neon_c4 = vpaddq_u8(vreinterpretq_u8_u32(neon_b7), vreinterpretq_u8_u32(neon_b8));

        const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);
        const uint8x16_t neon_d2 = vpaddq_u8(neon_c3, neon_c4);

        vst1q_u8(out + l - multi_loop, neon_d1);
        vst1q_u8(out + l - multi_loop + 16, neon_d2);
    } else {
        l = 0;
    }
    if (n & double_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);

        const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);
        const uint32x4_t neon_b3 = vcvtaq_u32_f32(neon_a3);
        const uint32x4_t neon_b4 = vcvtaq_u32_f32(neon_a4);

        const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
        const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));

        const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);

        vst1q_u8(out + l, neon_d1);
        l += double_loop;
    }
    if (n & single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);

        const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);

        const uint16x8_t neon_c1 = vpaddq_u16(vreinterpretq_u16_u32(neon_b1), vreinterpretq_u16_u32(neon_b2));

        const uint8x8_t neon_d1 = vmovn_u16(neon_c1);

        vst1_u8(out + l, neon_d1);
        l += single_loop;
    }
    for (; l < n; ++l) {
        out[l] = (uint8_t)(src[l] + 0.5);
    }
}

void quant_u8_with_parm(const float *src, idx_t n, uint8_t *out, float scale, float bias)
{
    idx_t l;
    const idx_t multi_loop = 32;
    const idx_t double_loop = 16;
    const idx_t single_loop = 8;
    float32x4_t neon_bias = vdupq_n_f32(bias);
    if (n >= multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src);
        float32x4_t neon_a2 = vld1q_f32(src + 4);
        float32x4_t neon_a3 = vld1q_f32(src + 8);
        float32x4_t neon_a4 = vld1q_f32(src + 12);
        float32x4_t neon_a5 = vld1q_f32(src + 16);
        float32x4_t neon_a6 = vld1q_f32(src + 20);
        float32x4_t neon_a7 = vld1q_f32(src + 24);
        float32x4_t neon_a8 = vld1q_f32(src + 28);
        for (l = multi_loop; l + multi_loop <= n; l += multi_loop) {
            neon_a1 = vmlaq_n_f32(neon_bias, neon_a1, scale);
            neon_a2 = vmlaq_n_f32(neon_bias, neon_a2, scale);
            neon_a3 = vmlaq_n_f32(neon_bias, neon_a3, scale);
            neon_a4 = vmlaq_n_f32(neon_bias, neon_a4, scale);
            neon_a5 = vmlaq_n_f32(neon_bias, neon_a5, scale);
            neon_a6 = vmlaq_n_f32(neon_bias, neon_a6, scale);
            neon_a7 = vmlaq_n_f32(neon_bias, neon_a7, scale);
            neon_a8 = vmlaq_n_f32(neon_bias, neon_a8, scale);

            const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
            const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);
            const uint32x4_t neon_b3 = vcvtaq_u32_f32(neon_a3);
            const uint32x4_t neon_b4 = vcvtaq_u32_f32(neon_a4);
            const uint32x4_t neon_b5 = vcvtaq_u32_f32(neon_a5);
            const uint32x4_t neon_b6 = vcvtaq_u32_f32(neon_a6);
            const uint32x4_t neon_b7 = vcvtaq_u32_f32(neon_a7);
            const uint32x4_t neon_b8 = vcvtaq_u32_f32(neon_a8);

            neon_a1 = vld1q_f32(src + l);
            neon_a2 = vld1q_f32(src + l + 4);
            neon_a3 = vld1q_f32(src + l + 8);
            neon_a4 = vld1q_f32(src + l + 12);
            neon_a5 = vld1q_f32(src + l + 16);
            neon_a6 = vld1q_f32(src + l + 20);
            neon_a7 = vld1q_f32(src + l + 24);
            neon_a8 = vld1q_f32(src + l + 28);

            const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
            const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));
            const uint8x16_t neon_c3 = vpaddq_u8(vreinterpretq_u8_u32(neon_b5), vreinterpretq_u8_u32(neon_b6));
            const uint8x16_t neon_c4 = vpaddq_u8(vreinterpretq_u8_u32(neon_b7), vreinterpretq_u8_u32(neon_b8));

            const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);
            const uint8x16_t neon_d2 = vpaddq_u8(neon_c3, neon_c4);

            vst1q_u8(out + l - multi_loop, neon_d1);
            vst1q_u8(out + l - multi_loop + 16, neon_d2);
        }
        neon_a1 = vmlaq_n_f32(neon_bias, neon_a1, scale);
        neon_a2 = vmlaq_n_f32(neon_bias, neon_a2, scale);
        neon_a3 = vmlaq_n_f32(neon_bias, neon_a3, scale);
        neon_a4 = vmlaq_n_f32(neon_bias, neon_a4, scale);
        neon_a5 = vmlaq_n_f32(neon_bias, neon_a5, scale);
        neon_a6 = vmlaq_n_f32(neon_bias, neon_a6, scale);
        neon_a7 = vmlaq_n_f32(neon_bias, neon_a7, scale);
        neon_a8 = vmlaq_n_f32(neon_bias, neon_a8, scale);

        const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);
        const uint32x4_t neon_b3 = vcvtaq_u32_f32(neon_a3);
        const uint32x4_t neon_b4 = vcvtaq_u32_f32(neon_a4);
        const uint32x4_t neon_b5 = vcvtaq_u32_f32(neon_a5);
        const uint32x4_t neon_b6 = vcvtaq_u32_f32(neon_a6);
        const uint32x4_t neon_b7 = vcvtaq_u32_f32(neon_a7);
        const uint32x4_t neon_b8 = vcvtaq_u32_f32(neon_a8);

        const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
        const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));
        const uint8x16_t neon_c3 = vpaddq_u8(vreinterpretq_u8_u32(neon_b5), vreinterpretq_u8_u32(neon_b6));
        const uint8x16_t neon_c4 = vpaddq_u8(vreinterpretq_u8_u32(neon_b7), vreinterpretq_u8_u32(neon_b8));

        const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);
        const uint8x16_t neon_d2 = vpaddq_u8(neon_c3, neon_c4);

        vst1q_u8(out + l - multi_loop, neon_d1);
        vst1q_u8(out + l - multi_loop + 16, neon_d2);
    } else {
        l = 0;
    }
    if (n & double_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        neon_a1 = vmlaq_n_f32(neon_bias, neon_a1, scale);
        neon_a2 = vmlaq_n_f32(neon_bias, neon_a2, scale);
        neon_a3 = vmlaq_n_f32(neon_bias, neon_a3, scale);
        neon_a4 = vmlaq_n_f32(neon_bias, neon_a4, scale);

        const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);
        const uint32x4_t neon_b3 = vcvtaq_u32_f32(neon_a3);
        const uint32x4_t neon_b4 = vcvtaq_u32_f32(neon_a4);

        const uint8x16_t neon_c1 = vpaddq_u8(vreinterpretq_u8_u32(neon_b1), vreinterpretq_u8_u32(neon_b2));
        const uint8x16_t neon_c2 = vpaddq_u8(vreinterpretq_u8_u32(neon_b3), vreinterpretq_u8_u32(neon_b4));

        const uint8x16_t neon_d1 = vpaddq_u8(neon_c1, neon_c2);

        vst1q_u8(out + l, neon_d1);
        l += double_loop;
    }
    if (n & single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        neon_a1 = vmlaq_n_f32(neon_bias, neon_a1, scale);
        neon_a2 = vmlaq_n_f32(neon_bias, neon_a2, scale);

        const uint32x4_t neon_b1 = vcvtaq_u32_f32(neon_a1);
        const uint32x4_t neon_b2 = vcvtaq_u32_f32(neon_a2);

        const uint16x8_t neon_c1 = vpaddq_u16(vreinterpretq_u16_u32(neon_b1), vreinterpretq_u16_u32(neon_b2));

        const uint8x8_t neon_d1 = vmovn_u16(neon_c1);

        vst1_u8(out + l, neon_d1);
        l += single_loop;
    }
    for (; l < n; ++l) {
        out[l] = (uint8_t)(src[l] * scale + bias + 0.5);
    }
}

void quant_s8(const float *src, idx_t n, int8_t *out)
{
    idx_t l = 0;
    const idx_t multi_loop = 32;
    const idx_t double_loop = 16;
    const idx_t single_loop = 8;
    for (; l + multi_loop <= n; l += multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        float32x4_t neon_a5 = vld1q_f32(src + l + 16);
        float32x4_t neon_a6 = vld1q_f32(src + l + 20);
        float32x4_t neon_a7 = vld1q_f32(src + l + 24);
        float32x4_t neon_a8 = vld1q_f32(src + l + 28);

        const int32x4_t neon_b1 = vcvtaq_s32_f32(neon_a1);
        const int32x4_t neon_b2 = vcvtaq_s32_f32(neon_a2);
        const int32x4_t neon_b3 = vcvtaq_s32_f32(neon_a3);
        const int32x4_t neon_b4 = vcvtaq_s32_f32(neon_a4);
        const int32x4_t neon_b5 = vcvtaq_s32_f32(neon_a5);
        const int32x4_t neon_b6 = vcvtaq_s32_f32(neon_a6);
        const int32x4_t neon_b7 = vcvtaq_s32_f32(neon_a7);
        const int32x4_t neon_b8 = vcvtaq_s32_f32(neon_a8);

        const int16x4_t neon_c1 = vmovn_s32(neon_b1);
        const int16x4_t neon_c3 = vmovn_s32(neon_b3);
        const int16x4_t neon_c5 = vmovn_s32(neon_b5);
        const int16x4_t neon_c7 = vmovn_s32(neon_b7);
        const int16x8_t neon_c2 = vmovn_high_s32(neon_c1, neon_b2);
        const int16x8_t neon_c4 = vmovn_high_s32(neon_c3, neon_b4);
        const int16x8_t neon_c6 = vmovn_high_s32(neon_c5, neon_b6);
        const int16x8_t neon_c8 = vmovn_high_s32(neon_c7, neon_b8);

        const int8x8_t neon_d1 = vmovn_s16(neon_c2);
        const int8x8_t neon_d3 = vmovn_s16(neon_c6);
        const int8x16_t neon_d2 = vmovn_high_s16(neon_d1, neon_c4);
        const int8x16_t neon_d4 = vmovn_high_s16(neon_d3, neon_c8);

        vst1q_s8(out + l, neon_d2);
        vst1q_s8(out + l + 16, neon_d4);
    }
    if (n & double_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);

        const int32x4_t neon_b1 = vcvtaq_s32_f32(neon_a1);
        const int32x4_t neon_b2 = vcvtaq_s32_f32(neon_a2);
        const int32x4_t neon_b3 = vcvtaq_s32_f32(neon_a3);
        const int32x4_t neon_b4 = vcvtaq_s32_f32(neon_a4);

        const int16x4_t neon_c1 = vmovn_s32(neon_b1);
        const int16x4_t neon_c3 = vmovn_s32(neon_b3);
        const int16x8_t neon_c2 = vmovn_high_s32(neon_c1, neon_b2);
        const int16x8_t neon_c4 = vmovn_high_s32(neon_c3, neon_b4);

        const int8x8_t neon_d1 = vmovn_s16(neon_c2);
        const int8x16_t neon_d2 = vmovn_high_s16(neon_d1, neon_c4);

        vst1q_s8(out + l, neon_d2);
        l += double_loop;
    }
    if (n & single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);

        const int32x4_t neon_b1 = vcvtaq_s32_f32(neon_a1);
        const int32x4_t neon_b2 = vcvtaq_s32_f32(neon_a2);

        const int16x4_t neon_c1 = vmovn_s32(neon_b1);
        const int16x8_t neon_c2 = vmovn_high_s32(neon_c1, neon_b2);

        const int8x8_t neon_d1 = vmovn_s16(neon_c2);

        vst1_s8(out + l, neon_d1);
        l += single_loop;
    }
    for (; l < n; ++l) {
        out[l] = (int8_t)(src[l] + (src[l] > 0 ? 0.5 : -0.5));
    }
}

void quant_s8_with_parm(const float *src, idx_t n, int8_t *out, float scale)
{
    idx_t l = 0;
    const idx_t multi_loop = 32;
    const idx_t double_loop = 16;
    const idx_t single_loop = 8;
    for (; l + multi_loop <= n; l += multi_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        float32x4_t neon_a5 = vld1q_f32(src + l + 16);
        float32x4_t neon_a6 = vld1q_f32(src + l + 20);
        float32x4_t neon_a7 = vld1q_f32(src + l + 24);
        float32x4_t neon_a8 = vld1q_f32(src + l + 28);
        neon_a1 = vmulq_n_f32(neon_a1, scale);
        neon_a2 = vmulq_n_f32(neon_a2, scale);
        neon_a3 = vmulq_n_f32(neon_a3, scale);
        neon_a4 = vmulq_n_f32(neon_a4, scale);
        neon_a5 = vmulq_n_f32(neon_a5, scale);
        neon_a6 = vmulq_n_f32(neon_a6, scale);
        neon_a7 = vmulq_n_f32(neon_a7, scale);
        neon_a8 = vmulq_n_f32(neon_a8, scale);

        const int32x4_t neon_b1 = vcvtnq_s32_f32(neon_a1);
        const int32x4_t neon_b2 = vcvtnq_s32_f32(neon_a2);
        const int32x4_t neon_b3 = vcvtnq_s32_f32(neon_a3);
        const int32x4_t neon_b4 = vcvtnq_s32_f32(neon_a4);
        const int32x4_t neon_b5 = vcvtnq_s32_f32(neon_a5);
        const int32x4_t neon_b6 = vcvtnq_s32_f32(neon_a6);
        const int32x4_t neon_b7 = vcvtnq_s32_f32(neon_a7);
        const int32x4_t neon_b8 = vcvtnq_s32_f32(neon_a8);

        const int16x4_t neon_c1 = vmovn_s32(neon_b1);
        const int16x4_t neon_c3 = vmovn_s32(neon_b3);
        const int16x4_t neon_c5 = vmovn_s32(neon_b5);
        const int16x4_t neon_c7 = vmovn_s32(neon_b7);
        const int16x8_t neon_c2 = vmovn_high_s32(neon_c1, neon_b2);
        const int16x8_t neon_c4 = vmovn_high_s32(neon_c3, neon_b4);
        const int16x8_t neon_c6 = vmovn_high_s32(neon_c5, neon_b6);
        const int16x8_t neon_c8 = vmovn_high_s32(neon_c7, neon_b8);

        const int8x8_t neon_d1 = vmovn_s16(neon_c2);
        const int8x8_t neon_d3 = vmovn_s16(neon_c6);
        const int8x16_t neon_d2 = vmovn_high_s16(neon_d1, neon_c4);
        const int8x16_t neon_d4 = vmovn_high_s16(neon_d3, neon_c8);

        vst1q_s8(out + l, neon_d2);
        vst1q_s8(out + l + 16, neon_d4);
    }
    if (n & double_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        neon_a1 = vmulq_n_f32(neon_a1, scale);
        neon_a2 = vmulq_n_f32(neon_a2, scale);
        neon_a3 = vmulq_n_f32(neon_a3, scale);
        neon_a4 = vmulq_n_f32(neon_a4, scale);

        const int32x4_t neon_b1 = vcvtnq_s32_f32(neon_a1);
        const int32x4_t neon_b2 = vcvtnq_s32_f32(neon_a2);
        const int32x4_t neon_b3 = vcvtnq_s32_f32(neon_a3);
        const int32x4_t neon_b4 = vcvtnq_s32_f32(neon_a4);

        const int16x4_t neon_c1 = vmovn_s32(neon_b1);
        const int16x4_t neon_c3 = vmovn_s32(neon_b3);
        const int16x8_t neon_c2 = vmovn_high_s32(neon_c1, neon_b2);
        const int16x8_t neon_c4 = vmovn_high_s32(neon_c3, neon_b4);

        const int8x8_t neon_d1 = vmovn_s16(neon_c2);
        const int8x16_t neon_d2 = vmovn_high_s16(neon_d1, neon_c4);

        vst1q_s8(out + l, neon_d2);
        l += double_loop;
    }
    if (n & single_loop) {
        float32x4_t neon_a1 = vld1q_f32(src + l);
        float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        neon_a1 = vmulq_n_f32(neon_a1, scale);
        neon_a2 = vmulq_n_f32(neon_a2, scale);

        const int32x4_t neon_b1 = vcvtnq_s32_f32(neon_a1);
        const int32x4_t neon_b2 = vcvtnq_s32_f32(neon_a2);

        const int16x4_t neon_c1 = vmovn_s32(neon_b1);
        const int16x8_t neon_c2 = vmovn_high_s32(neon_c1, neon_b2);

        const int8x8_t neon_d1 = vmovn_s16(neon_c2);

        vst1_s8(out + l, neon_d1);
        l += single_loop;
    }
    for (; l < n; ++l) {
        const float res = src[l] * scale;
        out[l] = (int8_t)(res + (res > 0 ? 0.5 : -0.5));
    }
}

size_t compute_quant_parm(idx_t n, const float *x, int metric_type, int range, float *scale, float *bias)
{
    float _max = x[0], _min = x[0];
    for (size_t i = 0; i < n; ++i) {
        if (x[i] > _max) {
            _max = x[i];
        } else if (x[i] < _min) {
            _min = x[i];
        }
    }
    if (metric_type == METRIC_L2) {
        int max_range = (int)(range * 0.9 + 0.5);
        int min_range = (int)(range * 0.1 + 0.5);
        if (_max > max_range && _max < range && _min < min_range && _min > -1) {
            *scale = 1.0;
            *bias = 0.0;
            return 0;
        }
        *scale = (float)(range - 1) / (_max - _min);
        *bias = -_min * (*scale);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        range >>= 1;
        int max_range = (int)(range * 0.9 + 0.5);
        if (_max > max_range && _max < range && _min < -max_range && _min > -range) {
            *scale = 1.0;
            return 0;
        }
        *scale = (float)(range - 1) / (_max > -_min ? _max : -_min);
    } else {
        return 0;
    }
    return 1;
}

void quant_sq8(idx_t n, const float *x, uint8_t *out, int metric_type, int use_parm, float scale, float bias)
{
    if (metric_type == METRIC_L2) {
        if (use_parm == 0) {
            quant_u8(x, n, out);
        } else {
            quant_u8_with_parm(x, n, out, scale, bias);
        }
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        if (use_parm == 0) {
            quant_s8(x, n, (int8_t *)out);
        } else {
            quant_s8_with_parm(x, n, (int8_t *)out, scale);
        }
    }
}