#ifndef STELLINE_UTILS_TENSOR_HH
#define STELLINE_UTILS_TENSOR_HH

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

}  // namespace stelline

#endif  // STELLINE_UTILS_TENSOR_HH
