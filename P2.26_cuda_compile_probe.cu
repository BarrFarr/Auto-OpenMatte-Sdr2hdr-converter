#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void identity_copy(const float* input, float* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}

int main() {
    constexpr std::size_t count = 256;
    constexpr std::size_t bytes = count * sizeof(float);

    float* input = nullptr;
    float* output = nullptr;
    cudaError_t status = cudaMallocManaged(&input, bytes);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged(input) failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    status = cudaMallocManaged(&output, bytes);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocManaged(output) failed: %s\n", cudaGetErrorString(status));
        cudaFree(input);
        return 1;
    }

    for (std::size_t index = 0; index < count; ++index) {
        input[index] = static_cast<float>(index) + 0.25f;
        output[index] = 0.0f;
    }

    identity_copy<<<(count + 127) / 128, 128>>>(input, output, count);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }

    bool correct = status == cudaSuccess;
    if (!correct) {
        std::fprintf(stderr, "CUDA execution failed: %s\n", cudaGetErrorString(status));
    } else {
        for (std::size_t index = 0; index < count; ++index) {
            if (output[index] != input[index]) {
                std::fprintf(stderr, "copy mismatch at %zu: expected %.2f, got %.2f\n", index, input[index], output[index]);
                correct = false;
                break;
            }
        }
    }

    cudaFree(output);
    cudaFree(input);
    if (correct) {
        std::puts("CUDA_IDENTITY_COPY=PASS");
        return 0;
    }
    return 1;
}
