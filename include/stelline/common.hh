#ifndef STELLINE_COMMON_HH
#define STELLINE_COMMON_HH

namespace stelline {

#define STELLINE_API __attribute__((visibility("default")))
#define STELLINE_HIDDEN __attribute__((visibility("hidden")))

#ifndef STELLINE_CUDA_CHECK_THROW
#define STELLINE_CUDA_CHECK_THROW(f, callback) { \
    const cudaError_t error = call; \
    if (err != cudaSuccess) { \
        callback(); \
        throw std::runtime_error("[CUDA] Error Code: " + std::string(cudaGetErrorString(err))); \
    } \
}
#endif  // STELLINE_CUDA_CHECK_THROW

}  // namespace stelline

#endif  // STELLINE_COMMON_HH
