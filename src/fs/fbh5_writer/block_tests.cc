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
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/logger.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include <stelline/config.hh>
#include <stelline/fbh5_writer/block.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct ScratchDirectory {
    std::filesystem::path path;

    ScratchDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "stelline-test-XXXXXX").string();
        REQUIRE(mkdtemp(pattern.data()) != nullptr);
        path = pattern;
    }

    ~ScratchDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

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

struct BackendGuard {
    ~BackendGuard() {
        Backend::DestroyAll();
    }
};

struct TextValue {
    std::string value;
    JST_SERDES(value);
};

struct RealValue {
    F64 value = 0.0;
    JST_SERDES(value);
};

struct IntegerValue {
    I64 value = 0;
    JST_SERDES(value);
};

void SetEnvironment(Flowgraph& flowgraph, const std::string& key, const std::string& value) {
    REQUIRE(flowgraph.environment().set(key, TextValue{value}) == Result::SUCCESS);
}

void SetEnvironment(Flowgraph& flowgraph, const std::string& key, const F64 value) {
    REQUIRE(flowgraph.environment().set(key, RealValue{value}) == Result::SUCCESS);
}

void SetEnvironment(Flowgraph& flowgraph, const std::string& key, const I64 value) {
    REQUIRE(flowgraph.environment().set(key, IntegerValue{value}) == Result::SUCCESS);
}

void SetObservation(Flowgraph& flowgraph) {
    const std::string prefix = "observatory.antenna.1a.tunings.a";
    SetEnvironment(flowgraph, "observation.antennas.0", std::string("1a"));
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
