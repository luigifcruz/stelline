#include <mutex>
#include <unordered_map>
#include <future>
#include <functional>
#include <memory>
#include <string>

#include "cyberbridge/holoscan.hh"

using namespace Jetstream;

namespace CyberBridge {

template<Device D, typename T>
std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan(Tensor<D, T>& tensor) {
    // Convert type.
    // NOTE: Looks like GXF still doesn't support complex numbers.

    nvidia::gxf::PrimitiveType type;

    if constexpr (std::is_same<T, F32>::value) {
        type = nvidia::gxf::PrimitiveType::kFloat32;
    } else {
        JST_ERROR("Unsupported data type.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Convert device.

    nvidia::gxf::MemoryStorageType device;

    if constexpr (D == Device::CPU) {
        device = nvidia::gxf::MemoryStorageType::kHost;
    } else if constexpr (D == Device::CUDA) {
        device = nvidia::gxf::MemoryStorageType::kDevice;
    } else {
        JST_ERROR("Unsupported device.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Convert shape.

    std::vector<I32> tmp;
    for (const auto& dim : tensor.shape()) {
        tmp.push_back(static_cast<I32>(dim));
    }
    nvidia::gxf::Shape shape(tmp);

    // Convert stride.

    const auto stride = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(T));

    // Convert data.

    void* data = static_cast<void*>(tensor.data());

    // Create deallocator.

    auto dealloc = [buffer = tensor](void*) {
        (void)buffer;
        return nvidia::gxf::Success;
    };

    // Create GXF tensor.

    auto tg = std::make_shared<nvidia::gxf::Tensor>();
    tg->wrapMemory(shape, type, sizeof(T), stride, device, data, dealloc);

    // Create DL tensor.

    auto maybe_dl_ctx = tg->toDLManagedTensorContext();

    if (!maybe_dl_ctx.has_value()) {
        JST_ERROR("Failed to convert GXF tensor to DLManagedTensor.");
        JST_CHECK_THROW(Result::ERROR);
    }

    return std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
}

template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CPU, U8>(Tensor<Device::CPU, U8>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CPU, F32>(Tensor<Device::CPU, F32>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CPU, CU8>(Tensor<Device::CPU, CU8>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CPU, CF32>(Tensor<Device::CPU, CF32>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CUDA, U8>(Tensor<Device::CUDA, U8>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CUDA, F32>(Tensor<Device::CUDA, F32>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CUDA, CU8>(Tensor<Device::CUDA, CU8>& tensor);
template std::shared_ptr<holoscan::Tensor> Holoscan::TensorToHoloscan<Device::CUDA, CF32>(Tensor<Device::CUDA, CF32>& tensor);

}  // namespace CyberBridge