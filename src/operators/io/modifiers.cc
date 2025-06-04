#include "modifiers.hh"

namespace stelline::operators::io {

template<typename T>
auto CreateTensor(auto& t) {
    const auto& tensor = matx::make_tensor<T>({
        t->shape()[2],
        t->shape()[0],
        t->shape()[1],
        t->shape()[3],
    }, matx::MATX_DEVICE_MEMORY);
    return std::make_shared<holoscan::Tensor>(tensor.ToDlPack());
}

cudaError DspBlockAlloc(const std::shared_ptr<holoscan::Tensor>& tensor,
                        std::shared_ptr<holoscan::Tensor>& permutedTensor) {
    switch (tensor->dtype().code) {
        case 2:  //kDLFloat:
            permutedTensor = CreateTensor<float>(tensor);
            break;
        case 5:  //kDLComplex:
            permutedTensor = CreateTensor<cuda::std::complex<float>>(tensor);
            break;
        default:
            HOLOSCAN_LOG_ERROR("DspBlockAlloc: Unsupported data type");
            return cudaErrorNotSupported;
    }

    return cudaSuccess;
}

}  // namespace stelline::operators::io
