#ifndef STELLINE_TYPES_HH
#define STELLINE_TYPES_HH

#include <memory>
#include <cstdint>

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>
#include <stelline/metadata.hh>

namespace stelline {

struct DspBlock {
   uint64_t timestamp;
   std::shared_ptr<holoscan::Tensor> tensor;

    void setMetadata(const DspBlock& other) {
        this->timestamp = other.timestamp;
    }

    void setData(const DspBlock& other) {
        this->tensor = other.tensor;
    }
};

struct InferenceBlock {
    DspBlock dspBlock;
    std::shared_ptr<holoscan::Tensor> tensor;

    void setMetadata(const InferenceBlock& other) {
    }

    void data(const InferenceBlock& other) {
        this->tensor = other.tensor;
    }
};

typedef std::tuple<std::shared_ptr<holoscan::Operator>, std::shared_ptr<holoscan::Operator>, std::shared_ptr<Metadata>> BitInterface;

}  // namespace stelline

#endif  // STELLINE_TYPES_HH
