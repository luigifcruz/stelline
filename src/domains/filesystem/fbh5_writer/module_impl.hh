#ifndef STELLINE_DOMAINS_FILESYSTEM_FBH5_WRITER_MODULE_IMPL_HH
#define STELLINE_DOMAINS_FILESYSTEM_FBH5_WRITER_MODULE_IMPL_HH

#include <chrono>

#include <hdf5.h>

extern "C" {
#include "filterbankc99.h"
}

#include <stelline/domains/filesystem/fbh5_writer/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kTimeAxis = 0;
constexpr U64 kBeamAxis = 1;
constexpr U64 kChannelAxis = 2;
constexpr U64 kIfAxis = 3;
constexpr const char* kSourceName = "Stelline";

struct Fbh5WriterImpl : public Module::Impl, public DynamicConfig<Fbh5Writer> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    F64 getBandwidthMBps();
    F64 getTotalDataWrittenMb() const;
    U64 getChunkCounter() const;

 protected:
    void updateBandwidth(const U64 deltaBytes);

    Shape expectedShape;
    DataType inputDataType = DataType::None;

    hid_t faplId = -1;
    bool faplOpen = false;

    filterbank_h5_file_t fbh5File = {};
    bool fileOpen = false;
    bool dataOpen = false;
    bool maskOpen = false;
    uint8_t* mask = nullptr;

    const void* registeredBuffer = nullptr;
    U64 registeredBufferSize = 0;
    bool bufferRegistered = false;

    Tools::Snapshot<U64> bytesWritten{0};
    Tools::Snapshot<U64> chunkCounter{0};
    I64 bytesSinceLastMeasurement = 0;
    Tools::Snapshot<F64> bandwidthMBps{0.0};
    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DOMAINS_FILESYSTEM_FBH5_WRITER_MODULE_IMPL_HH
