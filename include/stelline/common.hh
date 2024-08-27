#ifndef STELLINE_COMMON_HH
#define STELLINE_COMMON_HH

#include <cuda_runtime.h>

#include <cstdint>

namespace stelline {

#define STELLINE_API __attribute__((visibility("default")))
#define STELLINE_HIDDEN __attribute__((visibility("hidden")))

#define STELLINE_CUDA_CHECK_THROW(f, callback) { \
    const cudaError_t err = f; \
    if (err != cudaSuccess) { \
        callback(); \
        printf("[CUDA] Error Code: %s", cudaGetErrorString(err)); \
        throw std::runtime_error("CUDA error occurred."); \
    } \
}

#define STELLINE_IS_CUDA defined(__CUDACC__)
#define STELLINE_IS_NOT_CUDA !defined(__CUDACC__)

}  // namespace stelline

#endif  // STELLINE_COMMON_HH
