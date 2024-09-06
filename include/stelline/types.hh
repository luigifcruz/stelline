#ifndef STELLINE_TYPES_HH
#define STELLINE_TYPES_HH

#include <memory>
#include <cstdint>

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

namespace stelline {

struct DspBlock {
   uint64_t timestamp;
   std::shared_ptr<holoscan::Tensor> tensor;
};

struct InferenceBlock {
    DspBlock dspBlock;
    std::shared_ptr<holoscan::Tensor> tensor;
};

}  // namespace stelline

#endif  // STELLINE_TYPES_HH
