#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include <jetstream/backend/base.hh>

#include <jetstream/domains/core/duplicate/block.hh>
#include <jetstream/domains/core/multiply_constant/block.hh>
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/logger.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include <stelline/config.hh>
#include <stelline/fbh5_reader/block.hh>
#include <stelline/fbh5_writer/block.hh>

#include "module_impl.hh"
#include "helpers.hh"

using namespace Jetstream;
using namespace Jetstream::Testing;

namespace {

constexpr const char* kTestProvider = "stelline-test-no-hardware";

struct Fbh5WriterTestImpl final : public Modules::Fbh5WriterImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {};

JST_REGISTER_MODULE(Fbh5WriterTestImpl,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    kTestProvider);

bool HasStagingFile(const std::filesystem::path& path) {
    const auto prefix = path.filename().string() + ".";
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
        const auto name = entry.path().filename().string();
        if (name.starts_with(prefix) && name.ends_with(".partial")) {
            return true;
        }
    }
    return false;
}

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE) && defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
constexpr I64 kSyncTime = 1700000000;
constexpr F64 kSamplePeriod = 1e-3;
constexpr U64 kBlockDuration = 16;

void SetObservation(Flowgraph& flowgraph, const bool horizontal = true) {
    const std::string prefix = "observatory.antenna.1a.tunings.a";
    SetEnvironment(flowgraph, "observation.antennas.0", std::string("1a"));
    SetEnvironment(flowgraph, "observatory.name", std::string("ATA"));
    SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.source_name", std::string("TEST"));
    SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.ra", M_PI / 2.0);
    SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.dec", M_PI / 6.0);
    if (horizontal) {
        SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.az", M_PI);
        SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.el", M_PI / 3.0);
    }
    SetEnvironment(flowgraph, "instance.bands.len", I64{1});
    SetEnvironment(flowgraph, "instance.bands.0.tuning", std::string("a"));
    SetEnvironment(flowgraph, "instance.bands.0.band_index", I64{0});
    SetEnvironment(flowgraph, prefix + ".bands.0.frequency_start", 1000.0);
    SetEnvironment(flowgraph, prefix + ".bands.0.frequency_stop", 1002.0);
    SetEnvironment(flowgraph, prefix + ".bands.0.channel_start", I64{352});
    SetEnvironment(flowgraph, prefix + ".bands.0.channel_stop", I64{354});
    SetEnvironment(flowgraph, prefix + ".fengine.synctime", kSyncTime);
    SetEnvironment(flowgraph, prefix + ".fengine.sample_period", kSamplePeriod);
}

template<typename T>
T ReadIntegerAttribute(const hid_t dataset, const char* name, const hid_t type) {
    const hid_t attribute = H5Aopen(dataset, name, H5P_DEFAULT);
    REQUIRE(attribute >= 0);
    T value = 0;
    REQUIRE(H5Aread(attribute, type, &value) >= 0);
    REQUIRE(H5Aclose(attribute) >= 0);
    return value;
}

F64 ReadAttribute(const hid_t dataset, const char* name) {
    const hid_t attribute = H5Aopen(dataset, name, H5P_DEFAULT);
    REQUIRE(attribute >= 0);
    F64 value = 0.0;
    REQUIRE(H5Aread(attribute, H5T_NATIVE_DOUBLE, &value) >= 0);
    REQUIRE(H5Aclose(attribute) >= 0);
    return value;
}
#endif

}  // namespace

TEST_CASE("FBH5 writer block creates its inert child module",
          "[stelline][fbh5_writer][block][lifecycle]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {1, 2, 3, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}) == Result::SUCCESS);

    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input"].requested("source", "buffer");

    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath.clear();
    writerConfig.recording = false;
    REQUIRE(flowgraph.blockCreate("writer",
                                  writerConfig,
                                  inputs,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  kTestProvider) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    REQUIRE(writer.state == Block::State::Created);
    REQUIRE(writer.outputs.empty());

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}

TEST_CASE("FBH5 writer block reports its incomplete reason",
          "[stelline][fbh5_writer][block][lifecycle][diagnostic]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {1, 2, 3, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}) == Result::SUCCESS);

    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input"].requested("source", "buffer");

    const std::filesystem::path missingParent = "stelline_test_missing_fbh5_parent";
    REQUIRE_FALSE(std::filesystem::exists(missingParent));

    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath = (missingParent / "file.fbh5").string();
    writerConfig.recording = true;

    JST_LOG_LAST_ERROR() = "stale error";
    REQUIRE(flowgraph.blockCreate("writer",
                                  writerConfig,
                                  inputs,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  kTestProvider) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    REQUIRE(writer.state == Block::State::Incomplete);
    REQUIRE(writer.diagnostic ==
            "[MODULE_FBH5_WRITER] Parent directory 'stelline_test_missing_fbh5_parent' does not exist.");

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}

TEST_CASE("FBH5 writer waits for F-engine metadata before recording",
          "[stelline][fbh5_writer][block][lifecycle][metadata]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {1, 1, 2, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}) == Result::SUCCESS);

    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("duration", U64{16}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input"].requested("source", "buffer");

    const ScratchDirectory scratch;
    const auto path = scratch.path / "metadata.fbh5";
    std::ofstream(path) << "previous recording";

    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath = path.string();
    writerConfig.overwrite = true;
    writerConfig.recording = true;
    REQUIRE(flowgraph.blockCreate("writer",
                                  writerConfig,
                                  inputs,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  kTestProvider) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    REQUIRE(writer.state == Block::State::Incomplete);
    REQUIRE(writer.diagnostic ==
            "[MODULE_FBH5_WRITER] Missing required 'instance.bands.len' environment value.");
    std::ifstream preserved(path);
    CHECK(std::string(std::istreambuf_iterator<char>(preserved), {}) == "previous recording");
    CHECK_FALSE(HasStagingFile(path));

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}

TEST_CASE("FBH5 writer records RF start time, sample time, and channel frequencies",
          "[stelline][fbh5_writer][block][timing][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    using Catch::Matchers::WithinAbs;

    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "header.fbh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {2, 1, 4, 1};
        sourceConfig.dataType = "F32";
        REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData source;
        REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
        Tensor sourceOutput = source.outputs.at("buffer").tensor;
        REQUIRE(SetSignalAxes(sourceOutput, {
            .sample = Index{0},
            .channel = Index{2},
        }) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("duration", U64{4}) == Result::SUCCESS);

        TensorMap deviceInputs;
        deviceInputs["buffer"].requested("source", "buffer");
        Blocks::Duplicate deviceConfig;
        deviceConfig.outputDevice = "cuda";
        deviceConfig.hostAccessible = false;
        REQUIRE(flowgraph.blockCreate("device", deviceConfig, deviceInputs, DeviceType::CUDA) == Result::SUCCESS);

        TensorMap inputs;
        inputs["input"].requested("device", "buffer");

        Blocks::Fbh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.overwrite = true;
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        INFO(writer.diagnostic);
        REQUIRE(writer.state == Block::State::Created);

        REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);
        for (const U64 block : {U64{3}, U64{4}, U64{7}}) {
            REQUIRE(sourceOutput.setAttribute("timestamp", block * kBlockDuration) == Result::SUCCESS);
            REQUIRE(flowgraph.compute() == Result::SUCCESS);
            REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
            INFO(writer.diagnostic);
            REQUIRE(writer.state == Block::State::Created);
        }

        SetEnvironment(flowgraph, "observatory.antenna.1a.tunings.a.bands.0.frequency_start", 1000.5);
        REQUIRE(sourceOutput.setAttribute("timestamp", 8 * kBlockDuration) == Result::SUCCESS);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        REQUIRE(writer.state == Block::State::Errored);

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("device", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const hid_t data = H5Dopen(file, "data", H5P_DEFAULT);
    REQUIRE(data >= 0);
    const F64 tstart = ReadAttribute(data, "tstart");
    const F64 tsamp = ReadAttribute(data, "tsamp");
    const F64 fch1 = ReadAttribute(data, "fch1");
    const F64 foff = ReadAttribute(data, "foff");
    const F64 barycentric = ReadAttribute(data, "barycentric");
    const F64 pulsarcentric = ReadAttribute(data, "pulsarcentric");
    const F64 telescopeId = ReadAttribute(data, "telescope_id");
    const F64 rightAscension = ReadAttribute(data, "src_raj");
    const F64 declination = ReadAttribute(data, "src_dej");
    const F64 azimuth = ReadAttribute(data, "az_start");
    const F64 zenithAngle = ReadAttribute(data, "za_start");
    const hid_t sourceAttribute = H5Aopen(data, "source_name", H5P_DEFAULT);
    REQUIRE(sourceAttribute >= 0);
    const hid_t sourceType = H5Aget_type(sourceAttribute);
    REQUIRE(sourceType >= 0);
    std::string sourceName(H5Tget_size(sourceType), '\0');
    REQUIRE(H5Aread(sourceAttribute, sourceType, sourceName.data()) >= 0);
    REQUIRE(H5Tclose(sourceType) >= 0);
    REQUIRE(H5Aclose(sourceAttribute) >= 0);
    const U64 startSample = ReadIntegerAttribute<U64>(data, "rf_start_sample", H5T_NATIVE_UINT64);
    const U64 synctime = ReadIntegerAttribute<U64>(data, "synctime", H5T_NATIVE_UINT64);
    const U64 samplesPerRow = ReadIntegerAttribute<U64>(data, "rf_samples_per_row", H5T_NATIVE_UINT64);
    const hid_t space = H5Dget_space(data);
    REQUIRE(space >= 0);
    hsize_t dims[3] = {};
    REQUIRE(H5Sget_simple_extent_dims(space, dims, nullptr) == 3);
    REQUIRE(H5Sclose(space) >= 0);
    REQUIRE(H5Dclose(data) >= 0);
    REQUIRE(H5Fclose(file) >= 0);

    const F64 startSeconds = static_cast<F64>(kSyncTime) + 3.0 * kBlockDuration * kSamplePeriod;
    CHECK_THAT(tstart, WithinAbs(40587.0 + startSeconds / 86400.0, 1e-10));
    CHECK_THAT(tsamp, WithinAbs(kBlockDuration * kSamplePeriod / 2.0, 1e-15));
    CHECK_THAT(fch1, WithinAbs(1000.0, 1e-12));
    CHECK_THAT(foff, WithinAbs(0.5, 1e-12));
    CHECK(barycentric == 0.0);
    CHECK(pulsarcentric == 0.0);
    CHECK(telescopeId == 9.0);
    CHECK(sourceName.c_str() == std::string("TEST"));
    CHECK_THAT(rightAscension, WithinAbs(6.0, 1e-12));
    CHECK_THAT(declination, WithinAbs(30.0, 1e-12));
    CHECK_THAT(azimuth, WithinAbs(180.0, 1e-12));
    CHECK_THAT(zenithAngle, WithinAbs(30.0, 1e-12));
    CHECK(startSample == 3 * kBlockDuration);
    CHECK(synctime == static_cast<U64>(kSyncTime));
    CHECK(samplesPerRow == kBlockDuration / 2);
    CHECK(dims[0] == 10);
    CHECK(dims[2] == 4);
#endif
}

TEST_CASE("FBH5 writer keeps the destination when it cannot replace it",
          "[stelline][fbh5_writer][block][lifecycle][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "destination";
    REQUIRE(std::filesystem::create_directory(path));
    std::ofstream(path / "kept") << "kept";

    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    SetObservation(flowgraph);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {2, 1, 4, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);

    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);

    TensorMap deviceInputs;
    deviceInputs["buffer"].requested("source", "buffer");
    Blocks::Duplicate deviceConfig;
    deviceConfig.outputDevice = "cuda";
    deviceConfig.hostAccessible = false;
    REQUIRE(flowgraph.blockCreate("device", deviceConfig, deviceInputs, DeviceType::CUDA) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input"].requested("device", "buffer");

    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath = path.string();
    writerConfig.overwrite = true;
    writerConfig.recording = true;
    REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    INFO(writer.diagnostic);
    REQUIRE(writer.state == Block::State::Created);
    CHECK(std::filesystem::is_directory(path));
    CHECK(HasStagingFile(path));

    REQUIRE(sourceOutput.setAttribute("timestamp", kBlockDuration) == Result::SUCCESS);
    REQUIRE(flowgraph.compute() == Result::SUCCESS);
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    CHECK(writer.state == Block::State::Errored);
    CHECK(writer.diagnostic ==
          "[MODULE_FBH5_WRITER] Failed to move the new recording into '" + path.string() + "'.");

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("device", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);

    CHECK(std::filesystem::is_regular_file(path / "kept"));
    CHECK_FALSE(HasStagingFile(path));
#endif
}

TEST_CASE("FBH5 writer publishes nothing until a block is recorded",
          "[stelline][fbh5_writer][block][lifecycle][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "unpublished.fbh5";
    std::ofstream(path) << "previous recording";

    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    SetObservation(flowgraph);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {2, 1, 4, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);

    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);

    TensorMap deviceInputs;
    deviceInputs["buffer"].requested("source", "buffer");
    Blocks::Duplicate deviceConfig;
    deviceConfig.outputDevice = "cuda";
    deviceConfig.hostAccessible = false;
    REQUIRE(flowgraph.blockCreate("device", deviceConfig, deviceInputs, DeviceType::CUDA) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input"].requested("device", "buffer");

    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath = path.string();
    writerConfig.overwrite = true;
    writerConfig.recording = true;
    REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
    INFO(writer.diagnostic);
    REQUIRE(writer.state == Block::State::Created);

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("device", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);

    std::ifstream preserved(path);
    CHECK(std::string(std::istreambuf_iterator<char>(preserved), {}) == "previous recording");
    CHECK_FALSE(HasStagingFile(path));
#endif
}

TEST_CASE("FBH5 writer records without azimuth and elevation",
          "[stelline][fbh5_writer][block][timing][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "horizontal.fbh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph, false);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {2, 1, 4, 1};
        sourceConfig.dataType = "F32";
        REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData source;
        REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
        Tensor sourceOutput = source.outputs.at("buffer").tensor;
        REQUIRE(SetSignalAxes(sourceOutput, {
            .sample = Index{0},
            .channel = Index{2},
        }) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);

        TensorMap deviceInputs;
        deviceInputs["buffer"].requested("source", "buffer");
        Blocks::Duplicate deviceConfig;
        deviceConfig.outputDevice = "cuda";
        deviceConfig.hostAccessible = false;
        REQUIRE(flowgraph.blockCreate("device", deviceConfig, deviceInputs, DeviceType::CUDA) == Result::SUCCESS);

        TensorMap inputs;
        inputs["input"].requested("device", "buffer");

        Blocks::Fbh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        INFO(writer.diagnostic);
        REQUIRE(writer.state == Block::State::Created);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("device", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const hid_t data = H5Dopen(file, "data", H5P_DEFAULT);
    REQUIRE(data >= 0);
    CHECK(ReadAttribute(data, "az_start") == -1.0);
    CHECK(ReadAttribute(data, "za_start") == -1.0);
    CHECK_THAT(ReadAttribute(data, "src_raj"), Catch::Matchers::WithinAbs(6.0, 1e-12));
    REQUIRE(H5Dclose(data) >= 0);
    REQUIRE(H5Fclose(file) >= 0);
#endif
}

TEST_CASE("FBH5 reader returns what the writer recorded, including padded rows",
          "[stelline][fbh5_writer][fbh5_reader][block][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "roundtrip.fbh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {2, 1, 4, 1};
        sourceConfig.dataType = "F32";
        REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData source;
        REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
        Tensor sourceOutput = source.outputs.at("buffer").tensor;
        REQUIRE(SetSignalAxes(sourceOutput, {
            .sample = Index{0},
            .channel = Index{2},
        }) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);

        TensorMap deviceInputs;
        deviceInputs["buffer"].requested("source", "buffer");
        Blocks::Duplicate deviceConfig;
        deviceConfig.outputDevice = "cuda";
        deviceConfig.hostAccessible = false;
        REQUIRE(flowgraph.blockCreate("device", deviceConfig, deviceInputs, DeviceType::CUDA) == Result::SUCCESS);

        TensorMap inputs;
        inputs["input"].requested("device", "buffer");
        Blocks::Fbh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        for (const U64 block : {U64{0}, U64{1}, U64{3}}) {
            REQUIRE(sourceOutput.setAttribute("timestamp", block * kBlockDuration) == Result::SUCCESS);
            REQUIRE(flowgraph.compute() == Result::SUCCESS);
            REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
            INFO(writer.diagnostic);
            REQUIRE(writer.state == Block::State::Created);
        }

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("device", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    Blocks::Fbh5Reader readerConfig;
    readerConfig.filepath = path.string();
    readerConfig.batchSize = 2;
    readerConfig.loop = false;
    readerConfig.playing = true;
    REQUIRE(flowgraph.blockCreate("reader", readerConfig, {}, DeviceType::CPU) == Result::SUCCESS);

    Flowgraph::View::BlockData reader;
    REQUIRE(flowgraph.view().block("reader", reader) == Result::SUCCESS);
    INFO(reader.diagnostic);
    REQUIRE(reader.state == Block::State::Created);
    const Tensor signal = reader.outputs.at("signal").tensor;
    const Tensor mask = reader.outputs.at("mask").tensor;
    REQUIRE(signal.shape() == Shape{2, 1, 4});
    REQUIRE(mask.shape() == Shape{2, 1, 4});

    for (U64 block = 0; block < 4; ++block) {
        INFO("Block " << block);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        const bool padded = block == 2;
        for (U64 index = 0; index < signal.size(); ++index) {
            CHECK(signal.data<F32>()[index] == (padded ? 0.0f : 1.0f));
            CHECK(mask.data<U8>()[index] == (padded ? 1 : 0));
        }
    }

    REQUIRE(flowgraph.blockDestroy("reader", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
#endif
}

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE) && defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
namespace {

void RecordBand(const std::filesystem::path& path,
                const I64 bandIndex,
                const F64 frequencyStart,
                const I64 channelStart,
                const F32 value,
                const std::vector<U64>& blocks) {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    SetObservation(flowgraph);
    const std::string band = "observatory.antenna.1a.tunings.a.bands." + std::to_string(bandIndex);
    SetEnvironment(flowgraph, "instance.bands.0.band_index", bandIndex);
    SetEnvironment(flowgraph, band + ".frequency_start", frequencyStart);
    SetEnvironment(flowgraph, band + ".frequency_stop", frequencyStart + 2.0);
    SetEnvironment(flowgraph, band + ".channel_start", channelStart);
    SetEnvironment(flowgraph, band + ".channel_stop", channelStart + 2);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {2, 1, 4, 1};
    sourceConfig.dataType = "F32";
    REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);
    Flowgraph::View::BlockData source;
    REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
    Tensor sourceOutput = source.outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceOutput, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("timestamp", blocks.front() * kBlockDuration) == Result::SUCCESS);
    REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);

    TensorMap scaleInputs;
    scaleInputs["factor"].requested("source", "buffer");
    Blocks::MultiplyConstant scaleConfig;
    scaleConfig.constant = value;
    REQUIRE(flowgraph.blockCreate("scale", scaleConfig, scaleInputs, DeviceType::CUDA) == Result::SUCCESS);

    TensorMap deviceInputs;
    deviceInputs["buffer"].requested("scale", "product");
    Blocks::Duplicate deviceConfig;
    deviceConfig.outputDevice = "cuda";
    deviceConfig.hostAccessible = false;
    REQUIRE(flowgraph.blockCreate("device", deviceConfig, deviceInputs, DeviceType::CUDA) == Result::SUCCESS);

    TensorMap writerInputs;
    writerInputs["input"].requested("device", "buffer");
    Blocks::Fbh5Writer writerConfig;
    writerConfig.filepath = path.string();
    writerConfig.recording = true;
    REQUIRE(flowgraph.blockCreate("writer", writerConfig, writerInputs, DeviceType::CUDA) == Result::SUCCESS);

    Flowgraph::View::BlockData writer;
    for (const U64 block : blocks) {
        REQUIRE(sourceOutput.setAttribute("timestamp", block * kBlockDuration) == Result::SUCCESS);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        INFO(writer.diagnostic);
        REQUIRE(writer.state == Block::State::Created);
    }

    REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("device", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("scale", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}

template<typename T>
std::vector<T> ReadAll(const hid_t file, const char* name, const hid_t type) {
    const hid_t dataset = H5Dopen(file, name, H5P_DEFAULT);
    REQUIRE(dataset >= 0);
    const hid_t space = H5Dget_space(dataset);
    REQUIRE(space >= 0);
    std::vector<T> values(static_cast<size_t>(H5Sget_simple_extent_npoints(space)));
    REQUIRE(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    REQUIRE(H5Sclose(space) >= 0);
    REQUIRE(H5Dclose(dataset) >= 0);
    return values;
}

}  // namespace
#endif

TEST_CASE("Independently started band recordings stitch on one RF grid",
          "[stelline][fbh5_writer][block][timing][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    constexpr int kMissingDependency = 77;
    const int probe = RunProgram({STELLINE_TEST_PYTHON, "-c",
                                  "import sys\ntry:\n    import h5py, numpy\nexcept ImportError:\n    sys.exit(77)"});
    if (probe == kMissingDependency) {
        SKIP("The stitcher needs h5py and numpy.");
    }
    REQUIRE(probe == 0);

    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto lower = scratch.path / "lower.fbh5";
    const auto upper = scratch.path / "upper.fbh5";
    const auto joined = scratch.path / "joined.fbh5";
    RecordBand(lower, 0, 1000.0, 352, 1.0f, {2, 3, 4, 6, 7});
    RecordBand(upper, 1, 1002.0, 354, 2.0f, {0, 1, 2, 4, 5, 6});

    REQUIRE(RunProgram({STELLINE_TEST_PYTHON,
                        std::string(STELLINE_TEST_SOURCE_DIR) + "/tools/stitcher.py",
                        joined.string(),
                        upper.string(),
                        lower.string()}) == 0);

    const hid_t file = H5Fopen(joined.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const hid_t data = H5Dopen(file, "data", H5P_DEFAULT);
    REQUIRE(data >= 0);
    const U64 startSample = ReadIntegerAttribute<U64>(data, "rf_start_sample", H5T_NATIVE_UINT64);
    const F64 nchans = ReadAttribute(data, "nchans");
    const F64 fch1 = ReadAttribute(data, "fch1");
    REQUIRE(H5Dclose(data) >= 0);
    const auto values = ReadAll<F32>(file, "data", H5T_NATIVE_FLOAT);
    const auto mask = ReadAll<U8>(file, "mask", H5T_NATIVE_UINT8);
    REQUIRE(H5Fclose(file) >= 0);

    CHECK(startSample == 2 * kBlockDuration);
    CHECK(nchans == 8.0);
    CHECK_THAT(fch1, Catch::Matchers::WithinAbs(1000.0, 1e-12));

    constexpr U64 kRowsPerBlock = 2;
    constexpr U64 kChannels = 8;
    REQUIRE(values.size() == 5 * kRowsPerBlock * kChannels);
    REQUIRE(mask.size() == values.size());
    for (U64 row = 0; row < 5 * kRowsPerBlock; ++row) {
        const U64 block = 2 + row / kRowsPerBlock;
        for (U64 channel = 0; channel < kChannels; ++channel) {
            INFO("Block " << block << ", channel " << channel);
            const bool upperBand = channel >= 4;
            const bool missing = upperBand ? block == 3 : block == 5;
            const F32 expected = missing ? 0.0f : (upperBand ? 2.0f : 1.0f);
            CHECK(values[row * kChannels + channel] == expected);
            CHECK(mask[row * kChannels + channel] == (missing ? 1 : 0));
        }
    }
#endif
}

TEST_CASE("FBH5 writer arranges detector-ordered input without extra blocks",
          "[stelline][fbh5_writer][block][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "arranged.fbh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {1, 4, 2, 1};
        sourceConfig.dataType = "F32";
        REQUIRE(flowgraph.blockCreate("source", sourceConfig, {}, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData source;
        REQUIRE(flowgraph.view().block("source", source) == Result::SUCCESS);
        Tensor sourceOutput = source.outputs.at("buffer").tensor;
        REQUIRE(SetSignalAxes(sourceOutput, {
            .sample = Index{2},
            .channel = Index{1},
        }) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("timestamp", U64{0}) == Result::SUCCESS);
        REQUIRE(sourceOutput.setAttribute("duration", kBlockDuration) == Result::SUCCESS);

        TensorMap inputs;
        inputs["input"].requested("source", "buffer");
        Blocks::Fbh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        INFO(writer.diagnostic);
        REQUIRE(writer.state == Block::State::Created);

        for (const U64 block : {U64{0}, U64{1}}) {
            REQUIRE(sourceOutput.setAttribute("timestamp", block * kBlockDuration) == Result::SUCCESS);
            REQUIRE(flowgraph.compute() == Result::SUCCESS);
            REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
            INFO(writer.diagnostic);
            REQUIRE(writer.state == Block::State::Created);
        }

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const hid_t data = H5Dopen(file, "data", H5P_DEFAULT);
    REQUIRE(data >= 0);
    const F64 nchans = ReadAttribute(data, "nchans");
    const U64 samplesPerRow = ReadIntegerAttribute<U64>(data, "rf_samples_per_row", H5T_NATIVE_UINT64);
    const hid_t space = H5Dget_space(data);
    REQUIRE(space >= 0);
    hsize_t dims[3] = {};
    REQUIRE(H5Sget_simple_extent_dims(space, dims, nullptr) == 3);
    REQUIRE(H5Sclose(space) >= 0);
    REQUIRE(H5Dclose(data) >= 0);
    const auto values = ReadAll<F32>(file, "data", H5T_NATIVE_FLOAT);
    REQUIRE(H5Fclose(file) >= 0);

    CHECK(dims[0] == 4);
    CHECK(dims[1] == 1);
    CHECK(dims[2] == 4);
    CHECK(nchans == 4.0);
    CHECK(samplesPerRow == kBlockDuration / 2);
    CHECK(std::all_of(values.begin(), values.end(), [](const F32 value) { return value == 1.0f; }));
#endif
}
