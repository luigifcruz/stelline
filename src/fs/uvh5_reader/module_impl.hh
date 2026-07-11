#ifndef STELLINE_UVH5_READER_MODULE_IMPL_HH
#define STELLINE_UVH5_READER_MODULE_IMPL_HH

#include <chrono>

#include <hdf5.h>

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

    U64 getCurrentBlt() const;
    U64 getTotalBlts() const;
    F32 getCurrentBandwidth() const;

 protected:
    void updateBandwidth(const U64 deltaBytes);

    Tensor buffer;

    hid_t fileId = H5I_INVALID_HID;
    hid_t visDatasetId = H5I_INVALID_HID;
    hid_t visDataspaceId = H5I_INVALID_HID;

    // visdata dims: [Nblts, Nspws, Nfreqs, Npols] complex64.
    hsize_t dimNblts = 0;
    hsize_t dimNspws = 0;
    hsize_t dimNfreqs = 0;
    hsize_t dimNpols = 0;

    U64 currentBlt = 0;

    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};

    Tools::Snapshot<U64> snapshotTotalBlts{0};
    Tools::Snapshot<U64> snapshotCurrentBlt{0};
    Tools::Snapshot<F32> snapshotBandwidth{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_UVH5_READER_MODULE_IMPL_HH
