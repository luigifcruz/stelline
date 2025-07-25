#ifndef STELLINE_OPERATORS_FILESYSTEM_MODIFIERS_HH
#define STELLINE_OPERATORS_FILESYSTEM_MODIFIERS_HH

#include <stelline/common.hh>

#include <matx.h>
#include <cuda/std/complex>
#ifndef __CUDACC__
#include <holoscan/holoscan.hpp>
#endif  // __CUDACC__

namespace stelline::operators::filesystem {

#ifndef __CUDACC__
cudaError DspBlockAlloc(const std::shared_ptr<holoscan::Tensor>& tensor,
                        std::shared_ptr<holoscan::Tensor>& permutedTensor);
#endif  // __CUDACC__

cudaError_t DspBlockPermutation(DLManagedTensor* dst, const DLManagedTensor* src);
// TODO: Add DspBlockTypeCast method.

}  // namespace stelline::operators::filesystem

#endif // STELLINE_OPERATORS_FILESYSTEM_MODIFIERS_HH
