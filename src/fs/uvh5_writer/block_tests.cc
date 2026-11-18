#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <jetstream/backend/base.hh>
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include <stelline/config.hh>
#include <stelline/uvh5_reader/block.hh>
#include <stelline/uvh5_writer/block.hh>

#include "module_impl.hh"
#include "radiointerferometryc99.h"
#include "helpers.hh"

using namespace Jetstream;
using namespace Jetstream::Testing;

namespace {

constexpr const char* kTestProvider = "stelline-test-no-hardware";

struct Uvh5WriterTestImpl final : public Modules::Uvh5WriterImpl,
                                   public NativeCpuRuntimeContext,
                                   public Scheduler::Context {};

JST_REGISTER_MODULE(Uvh5WriterTestImpl,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    kTestProvider);

#if defined(STELLINE_DEVICE_CUDA_AVAILABLE) && defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
constexpr I64 kSyncTime = 1700000000;
constexpr F64 kSamplePeriod = 1e-3;
constexpr U64 kBlockDuration = 16;

constexpr F64 kLatitude = 40.8 * M_PI / 180.0;
constexpr F64 kLongitude = -121.5 * M_PI / 180.0;
constexpr F64 kAltitude = 1000.0;

void ExpectedApparent(const F64 icrsRightAscension,
                      const F64 icrsDeclination,
                      const F64 julianDate,
                      F64& rightAscension,
                      F64& declination) {
    double tai1 = 0.0, tai2 = 0.0, tt1 = 0.0, tt2 = 0.0;
    REQUIRE(eraUtctai(julianDate, 0.0, &tai1, &tai2) >= 0);
    REQUIRE(eraTaitt(tai1, tai2, &tt1, &tt2) == 0);
    double equationOfOrigins = 0.0;
    double cirsRightAscension = 0.0;
    eraAtci13(icrsRightAscension, icrsDeclination, 0.0, 0.0, 0.0, 0.0, tt1, tt2,
              &cirsRightAscension, &declination, &equationOfOrigins);
    rightAscension = eraAnp(cirsRightAscension - equationOfOrigins);
}

void SetObservation(Flowgraph& flowgraph) {
    SetEnvironment(flowgraph, "observatory.name", std::string("TEST"));
    SetEnvironment(flowgraph, "observatory.coordinates.latitude", 40.8);
    SetEnvironment(flowgraph, "observatory.coordinates.longitude", -121.5);
    SetEnvironment(flowgraph, "observatory.coordinates.altitude", 1000.0);
    SetEnvironment(flowgraph, "observation.antennas.length", I64{2});
    SetEnvironment(flowgraph, "observation.iers.pm_x_arcsec", 0.0);
    SetEnvironment(flowgraph, "observation.iers.pm_y_arcsec", 0.0);
    SetEnvironment(flowgraph, "observation.iers.ut1_utc", 0.0);
    SetEnvironment(flowgraph, "instance.bands.len", I64{1});
    SetEnvironment(flowgraph, "instance.bands.0.tuning", std::string("a"));
    SetEnvironment(flowgraph, "instance.bands.0.band_index", I64{0});

    const std::vector<std::string> antennas = {"1a", "1c"};
    for (size_t index = 0; index < antennas.size(); ++index) {
        const std::string prefix = "observatory.antenna." + antennas[index];
        SetEnvironment(flowgraph, "observation.antennas." + std::to_string(index), antennas[index]);
        SetEnvironment(flowgraph, prefix + ".number", static_cast<I64>(index + 1));
        SetEnvironment(flowgraph, prefix + ".diameter", 6.1);
        SetEnvironment(flowgraph, prefix + ".position.x", 10.0 * static_cast<F64>(index));
        SetEnvironment(flowgraph, prefix + ".position.y", 0.0);
        SetEnvironment(flowgraph, prefix + ".position.z", 0.0);
    }

    const std::string antenna = "observatory.antenna.1a";
    SetEnvironment(flowgraph, antenna + ".pointing.ra", 1.25);
    SetEnvironment(flowgraph, antenna + ".pointing.dec", 0.5);
    SetEnvironment(flowgraph, antenna + ".pointing.source_name", std::string("TEST"));
    SetEnvironment(flowgraph, antenna + ".tunings.a.bands.0.frequency_start", 1000.0);
    SetEnvironment(flowgraph, antenna + ".tunings.a.bands.0.frequency_stop", 1002.0);
    SetEnvironment(flowgraph, antenna + ".tunings.a.bands.0.channel_start", I64{0});
    SetEnvironment(flowgraph, antenna + ".tunings.a.bands.0.channel_stop", I64{2});
    SetEnvironment(flowgraph, antenna + ".tunings.a.fengine.synctime", kSyncTime);
    SetEnvironment(flowgraph, antenna + ".tunings.a.fengine.sample_period", kSamplePeriod);
}

std::string ReadText(const hid_t file, const char* path) {
    const hid_t dataset = H5Dopen(file, path, H5P_DEFAULT);
    REQUIRE(dataset >= 0);
    const hid_t type = H5Dget_type(dataset);
    REQUIRE(type >= 0);
    std::string value(H5Tget_size(type), '\0');
    REQUIRE(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data()) >= 0);
    REQUIRE(H5Tclose(type) >= 0);
    REQUIRE(H5Dclose(dataset) >= 0);
    return value.c_str();
}

template<typename T>
std::vector<T> ReadDataset(const hid_t file, const char* path) {
    const hid_t dataset = H5Dopen(file, path, H5P_DEFAULT);
    REQUIRE(dataset >= 0);
    const hid_t type = H5Dget_type(dataset);
    const hid_t space = H5Dget_space(dataset);
    REQUIRE(type >= 0);
    REQUIRE(space >= 0);
    REQUIRE(H5Tget_size(type) == sizeof(T));
    std::vector<T> values(static_cast<size_t>(H5Sget_simple_extent_npoints(space)));
    REQUIRE(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    REQUIRE(H5Sclose(space) >= 0);
    REQUIRE(H5Tclose(type) >= 0);
    REQUIRE(H5Dclose(dataset) >= 0);
    return values;
}
#endif

}  // namespace

TEST_CASE("UVH5 writer block creates its inert child module",
          "[stelline][uvh5_writer][block][lifecycle]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {1, 3, 4, 4};
    sourceConfig.dataType = "CF32";
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

    Blocks::Uvh5Writer writerConfig;
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

TEST_CASE("UVH5 writer pads an RF gap in the file and stops on an epoch change",
          "[stelline][uvh5_writer][block][timing][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "gap.uvh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {1, 3, 2, 4};
        sourceConfig.dataType = "CF32";
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

        TensorMap inputs;
        inputs["input"].requested("source", "buffer");

        Blocks::Uvh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.overwrite = true;
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        INFO(writer.diagnostic);
        REQUIRE(writer.state == Block::State::Created);

        for (const U64 block : {U64{0}, U64{1}, U64{4}}) {
            REQUIRE(sourceOutput.setAttribute("timestamp", block * kBlockDuration) == Result::SUCCESS);
            REQUIRE(flowgraph.compute() == Result::SUCCESS);
            REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
            INFO(writer.diagnostic);
            REQUIRE(writer.state == Block::State::Created);
        }

        SetEnvironment(flowgraph, "observatory.antenna.1a.tunings.a.fengine.synctime", kSyncTime + 1);
        REQUIRE(sourceOutput.setAttribute("timestamp", 5 * kBlockDuration) == Result::SUCCESS);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        REQUIRE(writer.state == Block::State::Errored);

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const auto ntimes = ReadDataset<I32>(file, "/Header/Ntimes");
    const auto times = ReadDataset<F64>(file, "/Header/time_array");
    const auto durations = ReadDataset<F64>(file, "/Header/integration_time");
    const auto flags = ReadDataset<H5_bool_t>(file, "/Data/flags");
    const auto weights = ReadDataset<F64>(file, "/Data/nsamples");
    const auto visibilities = ReadDataset<CF32>(file, "/Data/visdata");
    const auto frequencies = ReadDataset<F64>(file, "/Header/freq_array");
    const auto channelWidths = ReadDataset<F64>(file, "/Header/channel_width");
    const auto catalogRa = ReadDataset<F64>(file, "/Header/phase_center_catalog/0/cat_lon");
    const auto catalogDec = ReadDataset<F64>(file, "/Header/phase_center_catalog/0/cat_lat");
    const auto apparentRa = ReadDataset<F64>(file, "/Header/phase_center_app_ra");
    const auto apparentDec = ReadDataset<F64>(file, "/Header/phase_center_app_dec");
    const auto framePositionAngle = ReadDataset<F64>(file, "/Header/phase_center_frame_pa");
    const auto trackedUvw = ReadDataset<F64>(file, "/Header/uvw_array");
    const auto startSample = ReadDataset<U64>(file, "/Header/extra_keywords/rf_start_sample");
    const auto synctime = ReadDataset<U64>(file, "/Header/extra_keywords/synctime");
    const auto samplesPerRow = ReadDataset<U64>(file, "/Header/extra_keywords/rf_samples_per_row");
    REQUIRE(H5Fclose(file) >= 0);

    CHECK(catalogRa == std::vector<F64>{1.25});
    CHECK(catalogDec == std::vector<F64>{0.5});
    REQUIRE(apparentRa.size() == times.size());
    for (size_t row = 0; row < times.size(); ++row) {
        F64 expectedRa = 0.0;
        F64 expectedDec = 0.0;
        ExpectedApparent(1.25, 0.5, times[row], expectedRa, expectedDec);
        CHECK_THAT(apparentRa[row], Catch::Matchers::WithinAbs(expectedRa, 5e-6));
        CHECK_THAT(apparentDec[row], Catch::Matchers::WithinAbs(expectedDec, 5e-6));

        F64 southRa = 0.0, southDec = 0.0, northRa = 0.0, northDec = 0.0;
        ExpectedApparent(1.25, 0.5 - M_PI / 360.0, times[row], southRa, southDec);
        ExpectedApparent(1.25, 0.5 + M_PI / 360.0, times[row], northRa, northDec);
        CHECK_THAT(framePositionAngle[row], Catch::Matchers::WithinAbs(eraPas(southRa, southDec, northRa, northDec), 1e-6));
        CHECK(std::abs(framePositionAngle[row]) > 1e-4);

        double azimuth = 0.0, zenithDistance = 0.0, hourAngle = 0.0, declination = 0.0, observedRa = 0.0, origins = 0.0;
        REQUIRE(eraAtco13(1.25, 0.5, 0.0, 0.0, 0.0, 0.0, times[row], 0.0, 0.0,
                          kLongitude, kLatitude, kAltitude, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          &azimuth, &zenithDistance, &hourAngle, &declination, &observedRa, &origins) >= 0);
        const F64 elevation = M_PI / 2.0 - zenithDistance;
        const F64 east = -std::sin(kLongitude) * 10.0;
        const F64 north = -std::sin(kLatitude) * std::cos(kLongitude) * 10.0;
        const F64 up = std::cos(kLatitude) * std::cos(kLongitude) * 10.0;
        const F64 expectedW = east * std::sin(azimuth) * std::cos(elevation) +
                              north * std::cos(azimuth) * std::cos(elevation) +
                              up * std::sin(elevation);
        if (row % 3 == 0) {
            const F64* baseline = &trackedUvw[(row + 1) * 3];
            CHECK_THAT(baseline[2], Catch::Matchers::WithinAbs(expectedW, 1e-4));
            CHECK_THAT(baseline[0] * baseline[0] + baseline[1] * baseline[1],
                       Catch::Matchers::WithinAbs(100.0 - expectedW * expectedW, 1e-3));
        }
    }
    CHECK(startSample == std::vector<U64>{0});
    CHECK(synctime == std::vector<U64>{static_cast<U64>(kSyncTime)});
    CHECK(samplesPerRow == std::vector<U64>{kBlockDuration});

    constexpr size_t kBaselines = 3;
    constexpr size_t kSamplesPerIntegration = kBaselines * 2 * 4;
    REQUIRE(ntimes == std::vector<I32>{5});
    REQUIRE(frequencies == std::vector<F64>{1000.5e6, 1001.5e6});
    REQUIRE(channelWidths == std::vector<F64>{1e6, 1e6});
    REQUIRE(times.size() == 5 * kBaselines);
    REQUIRE(flags.size() == 5 * kSamplesPerIntegration);
    REQUIRE(weights.size() == flags.size());
    REQUIRE(visibilities.size() == flags.size());

    for (size_t integration = 0; integration < 5; ++integration) {
        INFO("Integration: " << integration);
        const bool missing = integration == 2 || integration == 3;
        const F64 seconds = static_cast<F64>(kSyncTime) +
                            (static_cast<F64>(integration * kBlockDuration) + kBlockDuration / 2.0) * kSamplePeriod;
        const F64 julianDate = 2440587.5 + seconds / 86400.0;
        for (size_t baseline = 0; baseline < kBaselines; ++baseline) {
            const size_t row = integration * kBaselines + baseline;
            CHECK_THAT(times[row], Catch::Matchers::WithinAbs(julianDate, 2e-9));
            CHECK_THAT(durations[row], Catch::Matchers::WithinAbs(kBlockDuration * kSamplePeriod, 1e-12));
        }
        for (size_t sample = 0; sample < kSamplesPerIntegration; ++sample) {
            const size_t index = integration * kSamplesPerIntegration + sample;
            CHECK(flags[index] == (missing ? H5_TRUE : H5_FALSE));
            CHECK(weights[index] == (missing ? 0.0 : static_cast<F64>(kSamplesPerIntegration)));
            CHECK(visibilities[index] == (missing ? CF32{} : CF32{1.0f, 0.0f}));
        }
    }
#endif
}

TEST_CASE("UVH5 writer infers fine channels per coarse channel from the input",
          "[stelline][uvh5_writer][block][frequency][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "channels.uvh5";

    const auto record = [&](const U64 channels) {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {1, 3, channels, 4};
        sourceConfig.dataType = "CF32";
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

        TensorMap inputs;
        inputs["input"].requested("source", "buffer");

        Blocks::Uvh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.overwrite = true;
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        const auto state = writer.state;
        const auto diagnostic = writer.diagnostic;
        if (state == Block::State::Created) {
            REQUIRE(flowgraph.compute() == Result::SUCCESS);
        }

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
        return std::pair{state, diagnostic};
    };

    std::ofstream(path) << "previous recording";
    const auto [mismatchState, mismatchDiagnostic] = record(3);
    CHECK(mismatchState == Block::State::Errored);
    CHECK(mismatchDiagnostic ==
          "[MODULE_UVH5_WRITER] Input channel count 3 does not split the 2 F-engine channels of the instance band.");
    {
        std::ifstream preserved(path);
        CHECK(std::string(std::istreambuf_iterator<char>(preserved), {}) == "previous recording");
    }

    const auto [state, diagnostic] = record(4);
    INFO(diagnostic);
    REQUIRE(state == Block::State::Created);

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const auto nfreqs = ReadDataset<I32>(file, "/Header/Nfreqs");
    const auto frequencies = ReadDataset<F64>(file, "/Header/freq_array");
    const auto channelWidths = ReadDataset<F64>(file, "/Header/channel_width");
    REQUIRE(H5Fclose(file) >= 0);

    CHECK(nfreqs == std::vector<I32>{4});
    CHECK(frequencies == std::vector<F64>{1000.0e6, 1000.5e6, 1001.0e6, 1001.5e6});
    CHECK(channelWidths == std::vector<F64>{0.5e6, 0.5e6, 0.5e6, 0.5e6});
#endif
}

TEST_CASE("UVH5 writer phases parked antennas to a fixed azimuth and elevation",
          "[stelline][uvh5_writer][block][timing][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    constexpr F64 kAzimuth = 2.356;
    constexpr F64 kElevation = 0.384;
    const ScratchDirectory scratch;
    const auto path = scratch.path / "parked.uvh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);
        SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.source_name", std::string("AzEl"));
        SetEnvironment(flowgraph, "observatory.antenna.1c.position.x", 0.0);
        SetEnvironment(flowgraph, "observatory.antenna.1c.position.y", 10.0);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {1, 3, 2, 4};
        sourceConfig.dataType = "CF32";
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

        TensorMap inputs;
        inputs["input"].requested("source", "buffer");

        Blocks::Uvh5Writer writerConfig;
        writerConfig.filepath = path.string();
        writerConfig.recording = true;
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);

        Flowgraph::View::BlockData writer;
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        REQUIRE(writer.state == Block::State::Incomplete);
        REQUIRE(writer.diagnostic ==
                "[MODULE_UVH5_WRITER] Missing required 'observatory.antenna.1a.pointing.az' environment value for a parked antenna.");
        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);

        SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.az", kAzimuth);
        SetEnvironment(flowgraph, "observatory.antenna.1a.pointing.el", kElevation);
        REQUIRE(flowgraph.blockCreate("writer", writerConfig, inputs, DeviceType::CUDA) == Result::SUCCESS);
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        INFO(writer.diagnostic);
        REQUIRE(writer.state == Block::State::Created);

        for (const U64 block : {U64{0}, U64{1}}) {
            REQUIRE(sourceOutput.setAttribute("timestamp", block * kBlockDuration) == Result::SUCCESS);
            REQUIRE(flowgraph.compute() == Result::SUCCESS);
        }
        REQUIRE(flowgraph.view().block("writer", writer) == Result::SUCCESS);
        REQUIRE(writer.state == Block::State::Created);

        REQUIRE(flowgraph.blockDestroy("writer", false) == Result::SUCCESS);
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const auto type = ReadText(file, "/Header/phase_center_catalog/0/cat_type");
    const auto frame = ReadText(file, "/Header/phase_center_catalog/0/cat_frame");
    const auto name = ReadText(file, "/Header/phase_center_catalog/0/cat_name");
    const auto longitude = ReadDataset<F64>(file, "/Header/phase_center_catalog/0/cat_lon");
    const auto latitude = ReadDataset<F64>(file, "/Header/phase_center_catalog/0/cat_lat");
    const auto times = ReadDataset<F64>(file, "/Header/time_array");
    const auto rightAscension = ReadDataset<F64>(file, "/Header/phase_center_app_ra");
    const auto declination = ReadDataset<F64>(file, "/Header/phase_center_app_dec");
    const auto positionAngle = ReadDataset<F64>(file, "/Header/phase_center_frame_pa");
    const auto uvw = ReadDataset<F64>(file, "/Header/uvw_array");
    REQUIRE(H5Fclose(file) >= 0);

    CHECK(type == "driftscan");
    CHECK(frame == "altaz");
    CHECK(name == "AzEl");
    CHECK(longitude == std::vector<F64>{kAzimuth});
    CHECK(latitude == std::vector<F64>{kElevation});

    constexpr size_t kBaselines = 3;
    REQUIRE(times.size() == 2 * kBaselines);
    REQUIRE(uvw.size() == 2 * kBaselines * 3);

    for (size_t integration = 0; integration < 2; ++integration) {
        const size_t row = integration * kBaselines;
        double icrsRightAscension = 0.0;
        double icrsDeclination = 0.0;
        REQUIRE(eraAtoc13("A", kAzimuth, M_PI / 2.0 - kElevation, times[row], 0.0, 0.0,
                          kLongitude, kLatitude, kAltitude, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          &icrsRightAscension, &icrsDeclination) >= 0);
        F64 expectedRightAscension = 0.0;
        F64 expectedDeclination = 0.0;
        ExpectedApparent(icrsRightAscension, icrsDeclination, times[row], expectedRightAscension, expectedDeclination);
        CHECK_THAT(rightAscension[row], Catch::Matchers::WithinAbs(expectedRightAscension, 5e-6));
        CHECK_THAT(declination[row], Catch::Matchers::WithinAbs(expectedDeclination, 5e-6));
        CHECK(positionAngle[row] == 0.0);
    }
    const F64 baselineY = 10.0;
    const F64 east = std::cos(kLongitude) * baselineY;
    const F64 north = -std::sin(kLatitude) * std::sin(kLongitude) * baselineY;
    const F64 up = std::cos(kLatitude) * std::sin(kLongitude) * baselineY;
    const F64 expectedW = east * std::sin(kAzimuth) * std::cos(kElevation) +
                          north * std::cos(kAzimuth) * std::cos(kElevation) +
                          up * std::sin(kElevation);
    CHECK_THAT(uvw[3 + 2], Catch::Matchers::WithinAbs(expectedW, 1e-3));
    CHECK(rightAscension[kBaselines] != rightAscension[0]);
    for (size_t index = 0; index < kBaselines * 3; ++index) {
        CHECK_THAT(uvw[kBaselines * 3 + index], Catch::Matchers::WithinAbs(uvw[index], 1e-12));
    }
#endif
}

TEST_CASE("UVH5 reader returns what the writer recorded, including padded integrations",
          "[stelline][uvh5_writer][uvh5_reader][block][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "roundtrip.uvh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {1, 3, 2, 4};
        sourceConfig.dataType = "CF32";
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

        TensorMap inputs;
        inputs["input"].requested("source", "buffer");
        Blocks::Uvh5Writer writerConfig;
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
        REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
    }

    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    Blocks::Uvh5Reader readerConfig;
    readerConfig.filepath = path.string();
    readerConfig.batchSize = 1;
    readerConfig.loop = false;
    readerConfig.playing = true;
    REQUIRE(flowgraph.blockCreate("reader", readerConfig, {}, DeviceType::CPU) == Result::SUCCESS);

    Flowgraph::View::BlockData reader;
    REQUIRE(flowgraph.view().block("reader", reader) == Result::SUCCESS);
    INFO(reader.diagnostic);
    REQUIRE(reader.state == Block::State::Created);
    const Tensor signal = reader.outputs.at("signal").tensor;
    REQUIRE(signal.shape() == Shape{1, 3, 2, 4});

    for (U64 integration = 0; integration < 4; ++integration) {
        INFO("Integration " << integration);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        const CF32 expected = integration == 2 ? CF32{} : CF32{1.0f, 0.0f};
        for (U64 index = 0; index < signal.size(); ++index) {
            CHECK(signal.data<CF32>()[index] == expected);
        }
    }

    REQUIRE(flowgraph.blockDestroy("reader", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
#endif
}

TEST_CASE("UVH5 writer arranges correlator-ordered input without extra blocks",
          "[stelline][uvh5_writer][block][cuda]") {
#if !defined(STELLINE_DEVICE_CUDA_AVAILABLE) || !defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    SKIP("Stelline was not built with CUDA support.");
#else
    BackendGuard backendGuard;
    Backend::Config backendConfig;
    backendConfig.headless = true;
    REQUIRE(Backend::Initialize<DeviceType::CPU>(backendConfig) == Result::SUCCESS);
    REQUIRE(Backend::Initialize<DeviceType::CUDA>(backendConfig) == Result::SUCCESS);

    const ScratchDirectory scratch;
    const auto path = scratch.path / "arranged.uvh5";

    {
        Flowgraph flowgraph;
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        SetObservation(flowgraph);

        Blocks::OnesTensor sourceConfig;
        sourceConfig.shape = {3, 2, 1, 4};
        sourceConfig.dataType = "CF32";
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
        Blocks::Uvh5Writer writerConfig;
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
    const auto ntimes = ReadDataset<I32>(file, "/Header/Ntimes");
    const auto nfreqs = ReadDataset<I32>(file, "/Header/Nfreqs");
    const auto visibilities = ReadDataset<CF32>(file, "/Data/visdata");
    REQUIRE(H5Fclose(file) >= 0);

    CHECK(ntimes == std::vector<I32>{2});
    CHECK(nfreqs == std::vector<I32>{2});
    CHECK(visibilities.size() == 2 * 3 * 2 * 4);
    CHECK(std::all_of(visibilities.begin(), visibilities.end(), [](const CF32 value) { return value == CF32{1.0f, 0.0f}; }));
#endif
}
