#ifndef STELLINE_UTILS_TENSOR_HH
#define STELLINE_UTILS_TENSOR_HH

#include <string>

#include <holoscan/holoscan.hpp>
#include <matx.h>

#include <stelline/yaml/types/block_shape.hh>

namespace stelline {

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
