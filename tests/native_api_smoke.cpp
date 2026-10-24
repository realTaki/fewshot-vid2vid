// CPU API probe only: not any repository CUDA kernel or GPU parity test.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>
int main() {
    auto input = at::ones({2, 2}, at::TensorOptions().dtype(at::kFloat));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "migration_cpu_probe", ([&] {
        const auto* values = input.data_ptr<scalar_t>();
        TORCH_CHECK(values[0] == scalar_t(1), "unexpected CPU tensor value");
    }));
    return 0;
}
