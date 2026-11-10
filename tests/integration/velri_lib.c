#include <velri_lib/linear.h>

/* Check the consumer header, exported CMake target and prepared-call interface.
 * Numerical edge cases belong to Velri-lib's independent correctness suite. */
int main(void) {
    const float input[] = {1.0f, 2.0f, 3.0f};
    const float weights[] = {4.0f, 5.0f, 6.0f, -1.0f, 0.0f, 1.0f};
    const float bias[] = {0.5f, -0.5f};
    float packed[6], output[2];
    velri_linear_plan plan;
    if (velri_linear_query(2, 3, VELRI_CPU_BASE, &plan) != VELRI_LINEAR_OK ||
        plan.packed_count != 6) return 1;
    if (velri_linear_pack(&plan, weights, 6, 3, packed, 6) != VELRI_LINEAR_OK) return 2;
    if (velri_linear_run(&plan, packed, 6, input, 3, 1, 3, bias, 2, output, 2, 2)
        != VELRI_LINEAR_OK) return 3;
    return output[0] == 32.5f && output[1] == 1.5f ? 0 : 4;
}
