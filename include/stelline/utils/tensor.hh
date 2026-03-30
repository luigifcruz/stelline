#ifndef STELLINE_UTILS_TENSOR_HH
#define STELLINE_UTILS_TENSOR_HH

#include <cstdint>
#include <string>

#include <holoscan/holoscan.hpp>
#include <matx.h>

#include <stelline/yaml/types/block_shape.hh>

namespace stelline {

inline int64_t TensorDataSizeBytes(const holoscan::Tensor& tensor) {
    int64_t element_bytes = tensor.dtype().bits / 8;

    if (tensor.dtype().lanes > 0) {
        element_bytes *= tensor.dtype().lanes;
    }
    if (tensor.dtype().code == 5) {
        element_bytes *= 2;
    }

    return tensor.size() * element_bytes;
}

template<typename T>
std::shared_ptr<holoscan::Tensor> MakeBlockTensor(const BlockShape& shape) {
    return std::make_shared<holoscan::Tensor>(
        matx::make_tensor<T>({
            static_cast<int64_t>(shape.numberOfAntennas),
            static_cast<int64_t>(shape.numberOfChannels),
            static_cast<int64_t>(shape.numberOfSamples),
            static_cast<int64_t>(shape.numberOfPolarizations)
        }, matx::MATX_DEVICE_MEMORY).ToDlPack());
}

inline std::shared_ptr<holoscan::Tensor> MakeBlockTensor(const BlockShape& shape, const std::string& dtype) {
    if (dtype == "cf32") {
        return MakeBlockTensor<cuda::std::complex<float>>(shape);
    }
    if (dtype == "ci8") {
        return MakeBlockTensor<cuda::std::complex<int8_t>>(shape);
    }
    throw std::runtime_error(fmt::format("Unsupported dtype: {}", dtype));
}

}  // namespace stelline

#endif  // STELLINE_UTILS_TENSOR_HH
