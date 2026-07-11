#include <algorithm>
#include <complex>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct Uvh5ReaderImplNativeCpu : public Uvh5ReaderImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;
};

Result Uvh5ReaderImplNativeCpu::create() {
    JST_CHECK(Uvh5ReaderImpl::create());
    return Result::SUCCESS;
}

Result Uvh5ReaderImplNativeCpu::computeSubmit() {
    if (fileId < 0 || !playing) {
        return Result::SUCCESS;
    }

    const U64 totalBlts = static_cast<U64>(dimNblts);

    U64 bltsRemaining = totalBlts - currentBlt;
    if (bltsRemaining == 0) {
        if (loop) {
            currentBlt = 0;
            bltsRemaining = totalBlts;
        } else {
            return Result::SUCCESS;
        }
    }

    const U64 bltsToRead = std::min(static_cast<U64>(batchSize), bltsRemaining);

    // Select hyperslab: [currentBlt, 0, 0, 0] -> [bltsToRead, Nspws, Nfreqs, Npols].
    const hsize_t start[4] = {static_cast<hsize_t>(currentBlt), 0, 0, 0};
    const hsize_t count[4] = {static_cast<hsize_t>(bltsToRead),
                              static_cast<hsize_t>(dimNspws),
                              static_cast<hsize_t>(dimNfreqs),
                              static_cast<hsize_t>(dimNpols)};

    if (H5Sselect_hyperslab(visDataspaceId, H5S_SELECT_SET,
                            start, nullptr, count, nullptr) < 0) {
        JST_ERROR("[MODULE_UVH5_READER] H5Sselect_hyperslab failed.");
        return Result::ERROR;
    }

    // HDF5 complex64 compound type matching std::complex<float> layout {r, i}.
    hid_t complexType = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
    H5Tinsert(complexType, "r", 0, H5T_NATIVE_FLOAT);
    H5Tinsert(complexType, "i", sizeof(float), H5T_NATIVE_FLOAT);

    hid_t memspace = H5Screate_simple(4, count, nullptr);
    if (memspace < 0) {
        H5Tclose(complexType);
        JST_ERROR("[MODULE_UVH5_READER] H5Screate_simple failed.");
        return Result::ERROR;
    }

    herr_t status = H5Dread(visDatasetId, complexType, memspace,
                            visDataspaceId, H5P_DEFAULT,
                            buffer.data<CF32>());
    H5Sclose(memspace);
    H5Tclose(complexType);

    if (status < 0) {
        JST_ERROR("[MODULE_UVH5_READER] H5Dread (visdata) failed at blt {}.", currentBlt);
        return Result::ERROR;
    }

    currentBlt += bltsToRead;
    snapshotCurrentBlt.publish(currentBlt);

    const U64 bytesRead = bltsToRead * static_cast<U64>(dimNspws) * static_cast<U64>(dimNfreqs) *
                          static_cast<U64>(dimNpols) * sizeof(std::complex<float>);
    updateBandwidth(bytesRead);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(Uvh5ReaderImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
