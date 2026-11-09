/* Compile this common C algorithm for each target. SIMD is across independent
 * outputs, never across the K reduction. Weights remain FP32 in
 * [ceil(M / C), K, C] order. Only compiler target flags/geometry differ.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

#ifndef VELRI_KERNEL
#error "VELRI_KERNEL must name this compiled variant"
#endif
#ifndef VELRI_COLUMNS
#error "VELRI_COLUMNS must describe the column tile"
#endif
#ifndef VELRI_PAIR_COLUMNS
#define VELRI_PAIR_COLUMNS VELRI_COLUMNS
#endif
_Static_assert(VELRI_PAIR_COLUMNS > 0 && VELRI_COLUMNS % VELRI_PAIR_COLUMNS == 0,
               "row-pair tile must divide the stored column tile");

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24, "FP32 is required");
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "FP64 is required");

/* FP32 finite products are exact in FP64 (at most 48 significant bits).
 * Contracting this product and its sequential FP64 addition therefore keeps
 * every finite-input result identical to separate multiply/add. No fast-math,
 * no multiple K accumulators, and no rounding of partial sums to FP32.
 */
#if defined(__clang__)
#pragma STDC FP_CONTRACT ON
#endif

static inline float finish(double sum, double bias) {
    float value = (float)(sum + bias);
    /* C arithmetic does not prescribe NaN payload propagation. A single
     * canonical output representation makes the C contract deterministic. */
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
        bits = UINT32_C(0x7fc00000);
        memcpy(&value, &bits, sizeof value);
    }
    return value;
}

void VELRI_KERNEL(const float *x, const float *packed, const float *bias, float *restrict y,
                  size_t rows, size_t m, size_t k, size_t x_stride, size_t y_stride) {
    for (size_t column = 0; column < m; column += VELRI_COLUMNS) {
        size_t row = 0;
        /* Reuse each load and FP32->FP64 conversion for two independent input
         * rows. K order within each output is unchanged. Single-token GEMV skips
         * this loop; the final odd GEMM row follows the same GEMV path. */
        for (; rows - row >= 2; row += 2) {
            for (size_t part = 0; part < VELRI_COLUMNS && part < m - column;
                 part += VELRI_PAIR_COLUMNS) {
                double sum0[VELRI_PAIR_COLUMNS], sum1[VELRI_PAIR_COLUMNS];
                for (size_t j = 0; j < VELRI_PAIR_COLUMNS; ++j)
                    sum0[j] = sum1[j] = -0.0;
                for (size_t i = 0; i < k; ++i) {
                    const double x0 = (double)x[row * x_stride + i];
                    const double x1 = (double)x[(row + 1) * x_stride + i];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                    for (size_t j = 0; j < VELRI_PAIR_COLUMNS; ++j) {
                        const double weight =
                            (double)packed[column * k + i * VELRI_COLUMNS + part + j];
                        sum0[j] += x0 * weight;
                        sum1[j] += x1 * weight;
                    }
                }
                const size_t left = m - column - part;
                const size_t width = left < VELRI_PAIR_COLUMNS ? left : VELRI_PAIR_COLUMNS;
                for (size_t j = 0; j < width; ++j) {
                    const double offset = bias ? (double)bias[column + part + j] : 0.0;
                    y[row * y_stride + column + part + j] = finish(sum0[j], offset);
                    y[(row + 1) * y_stride + column + part + j] = finish(sum1[j], offset);
                }
            }
        }
        for (; row < rows; ++row) {
            double sum[VELRI_COLUMNS];
            for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                sum[j] = -0.0;
            for (size_t i = 0; i < k; ++i) {
                const double input = (double)x[row * x_stride + i];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                    sum[j] += input * (double)packed[column * k + i * VELRI_COLUMNS + j];
            }
            const size_t width = m - column < VELRI_COLUMNS ? m - column : VELRI_COLUMNS;
            for (size_t j = 0; j < width; ++j)
                y[row * y_stride + column + j] =
                    finish(sum[j], bias ? (double)bias[column + j] : 0.0);
        }
    }
}
