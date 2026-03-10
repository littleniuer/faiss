#include "krl.h"
#include "krl_internal.h"
#include "platform_macros.h"
#include <stdlib.h>
#include <stdio.h>

static float distance_single_code_n8(const size_t M, const float *sim_table, const uint8_t *code, const float dis)
{
    const float *tab = sim_table;
    float result = dis;

    for (size_t m = 0; m < M; m++) {
        result += tab[*(code + m)];
        tab += 256;
    }

    return result;
}

static void distance_two_codes_n8(
    const size_t M, const float *sim_table, const uint8_t *__restrict codes, float *result, const float dis)
{
    const float *tab = sim_table;
    float result0 = dis;
    float result1 = dis;

    for (size_t m = 0; m < M; m++) {
        result0 += tab[*(codes + m)];
        result1 += tab[*(codes + M + m)];
        tab += 256;
    }
    result[0] = result0;
    result[1] = result1;
}

static void distance_four_codes_n8(
    const size_t M, const float *sim_table, const uint8_t *__restrict codes, float *result, const float dis)
{
    const float *tab = sim_table;
    float32x4_t res = vdupq_n_f32(dis);

    for (size_t m = 0; m < M; m++) {
        const float32x4_t neon_result = {
            tab[*(codes + m)], tab[*(codes + M + m)], tab[*(codes + 2 * M + m)], tab[*(codes + 3 * M + m)]};
        tab += 256;
        res = vaddq_f32(res, neon_result);
    }
    vst1q_f32(result, res);
}

static void distance_codes_n8_simd8(
    const size_t M, const float *sim_table, const uint8_t *__restrict codes, float *result, const float dis)
{
    const float *tab = sim_table;
    float32x4_t neon_result0 = vdupq_n_f32(dis);
    float32x4_t neon_result1 = vdupq_n_f32(dis);
    for (size_t m = 0; m < M; m++) {
        const float32x4_t neon_single_dim0 = {
            tab[*(codes + m)], tab[*(codes + M + m)], tab[*(codes + 2 * M + m)], tab[*(codes + 3 * M + m)]};
        const float32x4_t neon_single_dim1 = {
            tab[*(codes + 4 * M + m)], tab[*(codes + 5 * M + m)], tab[*(codes + 6 * M + m)], tab[*(codes + 7 * M + m)]};
        tab += 256;
        neon_result0 = vaddq_f32(neon_result0, neon_single_dim0);
        neon_result1 = vaddq_f32(neon_result1, neon_single_dim1);
    }
    vst1q_f32(result, neon_result0);
    vst1q_f32(result + 4, neon_result1);
}

static void distance_two_codes_idx_n8(const size_t M, const float *sim_table, const uint8_t *__restrict code0,
    const uint8_t *__restrict code1, float *result, const float dis)
{
    const float *tab = sim_table;
    float result0 = dis;
    float result1 = dis;

    for (size_t m = 0; m < M; m++) {
        result0 += tab[*(code0 + m)];
        result1 += tab[*(code1 + m)];
        tab += 256;
    }
    result[0] = result0;
    result[1] = result1;
}

static void distance_four_codes_idx_n8(
    const size_t M, const float *sim_table, const uint8_t *__restrict *codes, float *result, const float dis)
{
    const float *tab = sim_table;
    float result0 = dis;
    float result1 = dis;
    float result2 = dis;
    float result3 = dis;

    for (size_t m = 0; m < M; m++) {
        result0 += tab[*(codes[0] + m)];
        result1 += tab[*(codes[1] + m)];
        result2 += tab[*(codes[2] + m)];
        result3 += tab[*(codes[3] + m)];
        tab += 256;
    }
    result[0] = result0;
    result[1] = result1;
    result[2] = result2;
    result[3] = result3;
}

static void distance_eight_codes_idx_n8(
    const size_t M, const float *sim_table, const uint8_t *__restrict *codes, float *result, const float dis)
{
    const float *tab = sim_table;
    result[0] = dis;
    result[1] = dis;
    result[2] = dis;
    result[3] = dis;
    result[4] = dis;
    result[5] = dis;
    result[6] = dis;
    result[7] = dis;

    for (size_t m = 0; m < M; m++) {
        result[0] += tab[*(codes[0] + m)];
        result[1] += tab[*(codes[1] + m)];
        result[2] += tab[*(codes[2] + m)];
        result[3] += tab[*(codes[3] + m)];
        result[4] += tab[*(codes[4] + m)];
        result[5] += tab[*(codes[5] + m)];
        result[6] += tab[*(codes[6] + m)];
        result[7] += tab[*(codes[7] + m)];
        tab += 256;
    }
}

static void distance_codes_idx_n8_simd16(
    const size_t M, const float *sim_table, const uint8_t *__restrict *codes, float *result, const float dis)
{
    const float *tab = sim_table;
    float32x4_t neon_result0 = vdupq_n_f32(dis);
    float32x4_t neon_result1 = vdupq_n_f32(dis);
    float32x4_t neon_result2 = vdupq_n_f32(dis);
    float32x4_t neon_result3 = vdupq_n_f32(dis);
    for (size_t m = 0; m < M; m++) {
        const float32x4_t neon_single_dim0 = {
            tab[*(codes[0] + m)], tab[*(codes[1] + m)], tab[*(codes[2] + m)], tab[*(codes[3] + m)]};
        const float32x4_t neon_single_dim1 = {
            tab[*(codes[4] + m)], tab[*(codes[5] + m)], tab[*(codes[6] + m)], tab[*(codes[7] + m)]};
        const float32x4_t neon_single_dim2 = {
            tab[*(codes[8] + m)], tab[*(codes[9] + m)], tab[*(codes[10] + m)], tab[*(codes[11] + m)]};
        const float32x4_t neon_single_dim3 = {
            tab[*(codes[12] + m)], tab[*(codes[13] + m)], tab[*(codes[14] + m)], tab[*(codes[15] + m)]};
        tab += 256;
        neon_result0 = vaddq_f32(neon_result0, neon_single_dim0);
        neon_result1 = vaddq_f32(neon_result1, neon_single_dim1);
        neon_result2 = vaddq_f32(neon_result2, neon_single_dim2);
        neon_result3 = vaddq_f32(neon_result3, neon_single_dim3);
    }
    vst1q_f32(result, neon_result0);
    vst1q_f32(result + 4, neon_result1);
    vst1q_f32(result + 8, neon_result2);
    vst1q_f32(result + 12, neon_result3);
}

int krl_table_lookup_8b_f32(const size_t nsq, const size_t ncode, const uint8_t *codes, const float *sim_table,
    float *dis, const float dis0, size_t codes_size, size_t sim_table_size, size_t dis_size)
{
    size_t j = 0;
    for (; j + 8 <= ncode; j += 8) {
        distance_codes_n8_simd8(nsq, sim_table, codes + j * nsq, dis + j, dis0);
    }
    if (ncode & 4) {
        distance_four_codes_n8(nsq, sim_table, codes + j * nsq, dis + j, dis0);
        j += 4;
    }
    if (ncode & 2) {
        distance_two_codes_n8(nsq, sim_table, codes + j * nsq, dis + j, dis0);
    }
    if (ncode & 1) {
        dis[ncode - 1] = distance_single_code_n8(nsq, sim_table, codes + (ncode - 1) * nsq, dis0);
    }
    return SUCCESS;
}

int krl_table_lookup_8b_f32_by_idx(const size_t nsq, const size_t ncode, const uint8_t *codes, const float *sim_table,
    float *dis, const float dis0, const size_t *idx, size_t codes_size, size_t sim_table_size, size_t dis_size)
{
    const uint8_t *__restrict list_codes[16];

    size_t j = 0;
    for (; j + 16 <= ncode; j += 16) {
        list_codes[0] = codes + idx[j] * nsq;
        list_codes[1] = codes + idx[j + 1] * nsq;
        list_codes[2] = codes + idx[j + 2] * nsq;
        list_codes[3] = codes + idx[j + 3] * nsq;
        list_codes[4] = codes + idx[j + 4] * nsq;
        list_codes[5] = codes + idx[j + 5] * nsq;
        list_codes[6] = codes + idx[j + 6] * nsq;
        list_codes[7] = codes + idx[j + 7] * nsq;
        list_codes[8] = codes + idx[j + 8] * nsq;
        list_codes[9] = codes + idx[j + 9] * nsq;
        list_codes[10] = codes + idx[j + 10] * nsq;
        list_codes[11] = codes + idx[j + 11] * nsq;
        list_codes[12] = codes + idx[j + 12] * nsq;
        list_codes[13] = codes + idx[j + 13] * nsq;
        list_codes[14] = codes + idx[j + 14] * nsq;
        list_codes[15] = codes + idx[j + 15] * nsq;
        distance_codes_idx_n8_simd16(nsq, sim_table, list_codes, dis + j, dis0);
    }
    if (ncode & 8) {
        list_codes[0] = codes + idx[j] * nsq;
        list_codes[1] = codes + idx[j + 1] * nsq;
        list_codes[2] = codes + idx[j + 2] * nsq;
        list_codes[3] = codes + idx[j + 3] * nsq;
        list_codes[4] = codes + idx[j + 4] * nsq;
        list_codes[5] = codes + idx[j + 5] * nsq;
        list_codes[6] = codes + idx[j + 6] * nsq;
        list_codes[7] = codes + idx[j + 7] * nsq;
        distance_eight_codes_idx_n8(nsq, sim_table, list_codes, dis + j, dis0);
        j += 8;
    }
    if (ncode & 4) {
        list_codes[0] = codes + idx[j] * nsq;
        list_codes[1] = codes + idx[j + 1] * nsq;
        list_codes[2] = codes + idx[j + 2] * nsq;
        list_codes[3] = codes + idx[j + 3] * nsq;
        distance_four_codes_idx_n8(nsq, sim_table, list_codes, dis + j, dis0);
        j += 4;
    }
    if (ncode & 2) {
        const uint8_t *code1 = codes + idx[j] * nsq;
        const uint8_t *code2 = codes + idx[j + 1] * nsq;
        distance_two_codes_idx_n8(nsq, sim_table, code1, code2, dis + j, dis0);
    }
    if (ncode & 1) {
        dis[ncode - 1] = distance_single_code_n8(nsq, sim_table, codes + idx[ncode - 1] * nsq, dis0);
    }
    return SUCCESS;
}