#ifndef STELLINE_UVH5_READER_MODULE_IMPL_HH
#define STELLINE_UVH5_READER_MODULE_IMPL_HH

#include <chrono>

#include <hdf5.h>

extern "C" {
#include "uvh5.h"
}

#include <stelline/uvh5_reader/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct Uvh5ReaderImpl : public Module::Impl, public DynamicConfig<Uvh5Reader> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    U64 getCurrentBatchIndex() const;
    U64 getBatchCount() const;
    F32 getCurrentBandwidth() const;

 protected:
    void updateBandwidth(const U64 deltaBytes);

    Tensor buffer;

    hid_t faplId = H5P_DEFAULT;
    bool faplOpen = false;
    UVH5_file_t uvh5File = {0};

    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<U64> batchCount{0};
    Tools::Snapshot<U64> currentBatchIndex{0};
    Tools::Snapshot<F32> currentBandwidth{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_UVH5_READER_MODULE_IMPL_HH
