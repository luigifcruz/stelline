#ifndef STELLINE_UVH5_WRITER_MODULE_IMPL_HH
#define STELLINE_UVH5_WRITER_MODULE_IMPL_HH

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <hdf5.h>

extern "C" {
#include "uvh5.h"
}

#include <stelline/uvh5_writer/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kTimeAxis = 0;
constexpr U64 kBaselineAxis = 1;
constexpr U64 kFrequencyAxis = 2;
constexpr U64 kPolarizationAxis = 3;

constexpr U64 kExpectedTimeCount = 1;
constexpr U64 kExpectedPolarizationCount = 4;
constexpr const char* kObservationPolarization = "xy";

struct Uvh5AntennaInfo {
    std::string name;
    U64 number = 0;
    F64 diameter = 0.0;
    std::array<F64, 3> position = {0.0, 0.0, 0.0};
};

struct Uvh5Metadata {
    std::string telescopeName;
    F64 latitude = 0.0;
    F64 longitude = 0.0;
    F64 altitude = 0.0;

    std::vector<Uvh5AntennaInfo> antennas;

    std::string instanceTuning;
    U64 instanceBandIndex = 0;
    F64 instanceBandCenter = 0.0;
    F64 instanceBandwidth = 0.0;
    U64 instanceChannelCount = 0;

    F64 pointingRa = 0.0;
    F64 pointingDec = 0.0;
    std::string pointingSourceName;

    U64 packetTimestampOffset = 0;
    F64 sampleTimespan = 0.0;

    F64 iersPmXArcsec = 0.0;
    F64 iersPmYArcsec = 0.0;
    F64 iersUt1Utc = 0.0;

    bool valid = false;
};

struct Uvh5WriterImpl : public Module::Impl, public DynamicConfig<Uvh5Writer> {
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
    Result loadMetadata();
    Result refreshDynamicMetadata(const U64& timestamp);
    void updateBandwidth(const U64 deltaBytes);

    Shape expectedShape;
    DataType inputDataType = DataType::None;

    Uvh5Metadata metadata;

    hid_t faplId = -1;
    bool faplOpen = false;
    UVH5_file_t uvh5File = {};
    bool fileOpen = false;

    F64 integrationTimespan = 0.0;
    F64 frequencyBandwidth = 0.0;

    Tools::Snapshot<U64> bytesWritten{0};
    Tools::Snapshot<U64> chunkCounter{0};
    I64 bytesSinceLastMeasurement = 0;
    Tools::Snapshot<F64> bandwidthMBps{0.0};
    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_UVH5_WRITER_MODULE_IMPL_HH
