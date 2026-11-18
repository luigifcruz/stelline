#ifndef STELLINE_FBH5_WRITER_MODULE_IMPL_HH
#define STELLINE_FBH5_WRITER_MODULE_IMPL_HH

#include <chrono>
#include <filesystem>
#include <string>

#include <hdf5.h>

extern "C" {
#include "filterbankc99.h"
}

#include <stelline/fbh5_writer/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

#include "../recording_source.hh"
#include "../recording_timeline.hh"

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kTimeAxis = 0;
constexpr U64 kBeamAxis = 1;
constexpr U64 kChannelAxis = 2;
constexpr U64 kIfAxis = 3;
constexpr int kSigprocAtaTelescopeId = 9;
constexpr F64 kUnknownAngle = -1.0;

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
    Result loadHeaderMetadata(const Tensor& input);
    Result loadPointing();
    Result planBlock(const Tensor& input, RecordingGap& gap);
    Result commitBlock(const RecordingGap& gap);
    Result writeTiming(const BlockTiming& timing);
    Result appendChunks(const void* data, U64 samples, U64 missingBlocks);
    void updateBandwidth(const U64 deltaBytes);

    Shape validatedExpectedShape;
    DataType validatedInputDataType = DataType::None;

    Shape expectedShape;
    DataType inputDataType = DataType::None;

    hid_t faplId = -1;
    bool faplOpen = false;

    filterbank_h5_file_t fbh5File = {};
    bool openAttempted = false;
    bool fileOpen = false;
    bool dataOpen = false;
    bool maskOpen = false;
    uint8_t* mask = nullptr;
    RecordingTimeline timeline;
    bool blockWritten = false;

    RecordingSource source;
    std::filesystem::path partialPath;

    const void* registeredBuffer = nullptr;
    U64 registeredBufferSize = 0;
    bool bufferRegistered = false;

    Tools::Snapshot<U64> bytesWritten{0};
    Tools::Snapshot<U64> chunkCounter{0};
    I64 bytesSinceLastMeasurement = 0;
    Tools::Snapshot<F64> bandwidthMBps{0.0};
    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;

 private:
    Result appendChunk(const void* data);
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_FBH5_WRITER_MODULE_IMPL_HH
