// Register custom TTS ops with ExecuTorch runtime

#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <cstdio>
#include <vector>
#include <chrono>

namespace torch {
namespace executor {
namespace native {

// Forward declarations
executorch::aten::Tensor& istft_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& real,
    const executorch::aten::Tensor& imag,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    executorch::aten::Tensor& out);

executorch::aten::Tensor& layer_norm_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& input,
    executorch::runtime::ArrayRef<int64_t> normalized_shape,
    const executorch::aten::optional<executorch::aten::Tensor>& weight,
    const executorch::aten::optional<executorch::aten::Tensor>& bias,
    double eps,
    executorch::aten::Tensor& out);

} // namespace native
} // namespace executor
} // namespace torch

namespace {

// Wrapper function that unpacks EValues - signature must match OpFunction
void istft_wrapper(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> stack) {

    printf("[istft_wrapper] Called with %zu arguments\n", stack.size());

    // Debug: print the type of each argument
    for (size_t i = 0; i < stack.size(); ++i) {
        auto* ev = stack[i];
        const char* type_name = "unknown";
        if (ev->isTensor()) type_name = "Tensor";
        else if (ev->isInt()) type_name = "Int";
        else if (ev->isDouble()) type_name = "Double";
        else if (ev->isBool()) type_name = "Bool";
        else if (ev->isString()) type_name = "String";
        else if (ev->isNone()) type_name = "None";
        else if (ev->isIntList()) type_name = "IntList";
        else if (ev->isBoolList()) type_name = "BoolList";
        else if (ev->isDoubleList()) type_name = "DoubleList";
        else if (ev->isTensorList()) type_name = "TensorList";
        printf("  [%zu]: %s", i, type_name);
        if (ev->isInt()) printf(" = %lld", ev->toInt());
        if (ev->isTensor()) {
            auto& t = ev->toTensor();
            printf(" shape=[");
            for (int d = 0; d < t.dim(); ++d) {
                if (d > 0) printf(",");
                printf("%d", (int)t.size(d));
            }
            printf("]");
        }
        printf("\n");
    }

    // Updated: Handle 7 args if that's what we receive
    // Schema: istft.out(Tensor real, Tensor imag, int n_fft, int hop_length, int win_length, *, Tensor(a!) out) -> Tensor(a!)
    // If 7 args, there might be an extra optional argument (like a device or dtype)
    if (stack.size() < 6 || stack.size() > 7) {
        printf("[istft_wrapper] ERROR: Expected 6-7 args, got %zu\n", stack.size());
        ctx.fail(executorch::runtime::Error::InvalidArgument);
        return;
    }

    printf("[istft_wrapper] Unpacking arguments...\n");
    auto& real = stack[0]->toTensor();
    auto& imag = stack[1]->toTensor();
    int64_t n_fft = stack[2]->toInt();
    int64_t hop_length = stack[3]->toInt();
    int64_t win_length = stack[4]->toInt();
    // Stack[5] might be None (optional arg) or the out tensor
    // Stack[6] would be out tensor if [5] is None
    size_t out_idx = stack.size() == 7 ? 6 : 5;
    auto& out = stack[out_idx]->toTensor();

    printf("[istft_wrapper] real: [%d, %d, %d], n_fft=%lld, hop=%lld, win=%lld\n",
           (int)real.size(0), (int)real.size(1), (int)real.size(2),
           n_fft, hop_length, win_length);
    printf("[istft_wrapper] out: [%d, %d]\n",
           (int)out.size(0), (int)out.size(1));

    printf("[istft_wrapper] Calling istft_out...\n");
    torch::executor::native::istft_out(ctx, real, imag, n_fft, hop_length, win_length, out);
    printf("[istft_wrapper] Done\n");
}

// Wrapper for layer_norm.out
// Schema: layer_norm.out(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, *, Tensor(a!) out) -> Tensor(a!)
void layer_norm_wrapper(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> stack) {

    printf("[layer_norm_wrapper] Called with %zu arguments\n", stack.size());

    // Debug: print the type of each argument
    for (size_t i = 0; i < stack.size(); ++i) {
        auto* ev = stack[i];
        const char* type_name = "unknown";
        if (ev->isTensor()) type_name = "Tensor";
        else if (ev->isInt()) type_name = "Int";
        else if (ev->isDouble()) type_name = "Double";
        else if (ev->isBool()) type_name = "Bool";
        else if (ev->isNone()) type_name = "None";
        else if (ev->isIntList()) type_name = "IntList";
        printf("  [%zu]: %s\n", i, type_name);
    }

    // Expected: input, normalized_shape (int[]), weight (optional), bias (optional), eps, [optional extra], out
    if (stack.size() < 6 || stack.size() > 7) {
        printf("[layer_norm_wrapper] ERROR: Expected 6-7 args, got %zu\n", stack.size());
        ctx.fail(executorch::runtime::Error::InvalidArgument);
        return;
    }

    auto& input = stack[0]->toTensor();
    auto normalized_shape_list = stack[1]->toIntList();
    // Convert IntList to ArrayRef<int64_t>
    std::vector<int64_t> norm_shape_vec;
    for (size_t i = 0; i < normalized_shape_list.size(); ++i) {
        norm_shape_vec.push_back(normalized_shape_list[i]);
    }
    executorch::runtime::ArrayRef<int64_t> normalized_shape(norm_shape_vec.data(), norm_shape_vec.size());

    // Weight and bias are optional
    executorch::aten::optional<executorch::aten::Tensor> weight;
    executorch::aten::optional<executorch::aten::Tensor> bias;

    if (!stack[2]->isNone()) {
        weight = stack[2]->toTensor();
    }
    if (!stack[3]->isNone()) {
        bias = stack[3]->toTensor();
    }

    double eps = stack[4]->toDouble();
    // Out tensor is last arg (index 5 or 6 depending on if there's an extra optional)
    size_t out_idx = stack.size() - 1;
    auto& out = stack[out_idx]->toTensor();

    auto start = std::chrono::high_resolution_clock::now();
    torch::executor::native::layer_norm_out(ctx, input, normalized_shape, weight, bias, eps, out);
    auto end = std::chrono::high_resolution_clock::now();
    printf("[layer_norm_wrapper] Done in %.2f ms\n",
           std::chrono::duration<double, std::milli>(end - start).count());
}

// Register kernel at static initialization time
static executorch::runtime::Kernel custom_kernels[] = {
    executorch::runtime::Kernel(
        "tts::istft.out",
        istft_wrapper),
    executorch::runtime::Kernel(
        "tts::layer_norm.out",
        layer_norm_wrapper),
};

static auto register_kernels __attribute__((used)) =
    executorch::runtime::register_kernels({
        custom_kernels,
        sizeof(custom_kernels) / sizeof(custom_kernels[0])
    });

} // namespace
