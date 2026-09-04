#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear;
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        constexpr std::array invocations{
            Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
            Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{9, CallForm::Policy, ops::LinearPolicy::A16Only},
        };
        const int failures = run_shape(
            "FP8_BLOCK_A16", ActivationCompute::A16, make_fp8_block_weight,
            {512, 2560, 1973U, Comparison::Sampled, true, invocations});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " block FP8 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "block FP8 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
