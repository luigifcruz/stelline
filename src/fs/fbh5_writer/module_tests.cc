#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <vector>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_environment.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/fbh5_writer/module.hh>

#include "module_impl.hh"
#include "../helpers.hh"
#include "../recording_axes.hh"
#include "../recording_source.hh"

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

std::shared_ptr<Module> BuildModule() {
    const auto implementations = Registry::ListAvailableModules("fbh5_writer");
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fbh5_writer",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    return module;
}

TensorMap InputMap(Tensor tensor) {
    TensorMap inputs;
    inputs["input"].requested("test", "input");
    inputs["input"].tensor = std::move(tensor);
    return inputs;
}

Tensor WriterInput(const DataType dtype, const Shape& shape) {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, dtype, shape) == Result::SUCCESS);
    REQUIRE(SetSignalAxes(input, {
        .sample = Index{0},
        .channel = Index{2},
    }) == Result::SUCCESS);
    return input;
}

}  // namespace

TEST_CASE("FBH5 writer validates semantics before defining its interface",
          "[stelline][fbh5_writer][module][validation][lifecycle]") {
    auto module = BuildModule();
    auto input = WriterInput(DataType::F32, {1, 2, 3});

    Modules::Fbh5Writer config;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE(module->outputs().empty());
}

TEST_CASE("FBH5 writer delegates missing inputs to the framework",
          "[stelline][fbh5_writer][module][validation][inputs]") {
    auto module = BuildModule();

    Modules::Fbh5Writer config;
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs() ==
            Module::Interface::EntryList{"input"});
}

TEST_CASE("FBH5 writer provider rejects unsupported dtypes during validation",
          "[stelline][fbh5_writer][module][validation][dtype]") {
    auto module = BuildModule();
    auto input = WriterInput(DataType::F64, {1, 2, 3, 1});

    Modules::Fbh5Writer config;
    REQUIRE(module->create("test", config, InputMap(std::move(input))) ==
            Result::ERROR);
    REQUIRE(module->interface()->inputs().empty());
}

TEST_CASE("Recording gaps use the first written RF block and a fixed duration",
          "[stelline][fbh5_writer][module][timing]") {
    auto input = WriterInput(DataType::F32, {1, 1, 1, 1});
    Modules::RecordingTimeline timeline;
    Modules::RecordingGap gap;
    const auto plan = [&](const U64 timestamp, const U64 duration) {
        REQUIRE(input.setAttribute("timestamp", timestamp) == Result::SUCCESS);
        REQUIRE(input.setAttribute("duration", duration) == Result::SUCCESS);
        return timeline.plan(input, "[TEST]", gap);
    };

    REQUIRE(plan(8192, 8) == Result::SUCCESS);
    CHECK(gap.missingBlocks == 0);
    timeline.commit(gap);
    REQUIRE(plan(8200, 8) == Result::SUCCESS);
    CHECK(gap.missingBlocks == 0);
    timeline.commit(gap);
    REQUIRE(plan(8224, 8) == Result::SUCCESS);
    CHECK(gap.firstMissing == 8208);
    CHECK(gap.missingBlocks == 2);
    timeline.commit(gap);

    CHECK(plan(8224, 8) == Result::ERROR);
    CHECK(plan(8233, 8) == Result::ERROR);
    CHECK(plan(8232, 4) == Result::ERROR);
    CHECK(plan(8232, 0) == Result::ERROR);
    CHECK(plan(8232 + 1025 * 8, 8) == Result::ERROR);
    CHECK(plan(std::numeric_limits<U64>::max() - 7, 8) == Result::ERROR);
    REQUIRE(input.removeAttribute("duration") == Result::SUCCESS);
    CHECK(timeline.plan(input, "[TEST]", gap) == Result::ERROR);
    REQUIRE(plan(8232, 8) == Result::SUCCESS);
    CHECK(gap.missingBlocks == 0);

    Modules::RecordingTimeline fresh;
    REQUIRE(input.removeAttribute("duration") == Result::SUCCESS);
    CHECK(fresh.plan(input, "[TEST]", gap) == Result::ERROR);
    REQUIRE(input.setAttribute("duration", U64{8}) == Result::SUCCESS);
    REQUIRE(input.removeAttribute("timestamp") == Result::SUCCESS);
    CHECK(fresh.plan(input, "[TEST]", gap) == Result::ERROR);
}

TEST_CASE("FBH5 writes zero data and sets the mask only for missing chunks",
          "[stelline][fbh5_writer][module][timing]") {
    struct MemoryWriter : Modules::Fbh5WriterImpl {
        ~MemoryWriter() { (void)destroy(); }

        void open() {
            faplId = H5Pcreate(H5P_FILE_ACCESS);
            REQUIRE(faplId >= 0);
            faplOpen = true;
            REQUIRE(H5Pset_fapl_core(faplId, 65536, false) >= 0);
            fbh5File.header.nchans = 2;
            fbh5File.header.nifs = 1;
            fbh5File.nchans_per_write = 2;
            fbh5File.ntimes_per_write = 1;
            fbh5File.ds_data.filter_flag = H5_FILTER_FLAG_NONE;
            fbh5File.ds_mask.filter_flag = H5_FILTER_FLAG_NONE;
            openAttempted = true;
            REQUIRE(filterbank_h5_open_explicit("gap-test.fbh5", &fbh5File,
                                               H5Tcopy(H5T_IEEE_F32LE), faplId) >= 0);
            mask = static_cast<uint8_t*>(std::calloc(2, 1));
            REQUIRE(mask != nullptr);
        }

        using Fbh5WriterImpl::appendChunks;
        using Fbh5WriterImpl::fbh5File;
    };

    MemoryWriter writer;
    writer.open();
    const std::array<F32, 2> before = {2.0f, 3.0f};
    const std::array<F32, 2> after = {4.0f, 5.0f};
    REQUIRE(writer.appendChunks(before.data(), before.size(), 0) == Result::SUCCESS);
    REQUIRE(writer.appendChunks(after.data(), after.size(), 2) == Result::SUCCESS);

    std::array<F32, 8> data;
    std::array<U8, 8> mask;
    const auto& file = writer.fbh5File;
    REQUIRE(file.ds_data.dims[0] == 4);
    REQUIRE(file.ds_mask.dims[0] == 4);
    REQUIRE(H5Dread(file.ds_data.D_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) >= 0);
    REQUIRE(H5Dread(file.ds_mask.D_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask.data()) >= 0);
    CHECK(data == std::array<F32, 8>{2, 3, 0, 0, 0, 0, 4, 5});
    CHECK(mask == std::array<U8, 8>{0, 0, 1, 1, 1, 1, 0, 0});
}

TEST_CASE("Recording frequency axes center fine channels on the shifted FFT grid",
          "[stelline][fbh5_writer][module][frequency]") {
    using Catch::Matchers::WithinAbs;

    const auto coarse = Modules::ChannelFrequencies(1000.0, 1002.0, 2, 2);
    REQUIRE(coarse.has_value());
    CHECK_THAT(coarse->first, WithinAbs(1000.5, 1e-12));
    CHECK_THAT(coarse->step, WithinAbs(1.0, 1e-12));

    const auto fine = Modules::ChannelFrequencies(1000.0, 1002.0, 2, 8);
    REQUIRE(fine.has_value());
    CHECK_THAT(fine->first, WithinAbs(1000.0, 1e-12));
    CHECK_THAT(fine->step, WithinAbs(0.25, 1e-12));
    CHECK_THAT(fine->first + 2 * fine->step, WithinAbs(1000.5, 1e-12));
    CHECK_THAT(fine->first + 6 * fine->step, WithinAbs(1001.5, 1e-12));

    const auto adjacent = Modules::ChannelFrequencies(1002.0, 1004.0, 2, 8);
    REQUIRE(adjacent.has_value());
    CHECK_THAT(adjacent->first, WithinAbs(fine->first + 8 * fine->step, 1e-12));

    const auto inverted = Modules::ChannelFrequencies(1002.0, 1000.0, 2, 2);
    REQUIRE(inverted.has_value());
    CHECK_THAT(inverted->first, WithinAbs(1001.5, 1e-12));
    CHECK_THAT(inverted->step, WithinAbs(-1.0, 1e-12));

    CHECK_FALSE(Modules::ChannelFrequencies(1000.0, 1002.0, 2, 3).has_value());
    CHECK_FALSE(Modules::ChannelFrequencies(1000.0, 1002.0, 0, 2).has_value());
    CHECK_FALSE(Modules::ChannelFrequencies(1000.0, 1000.0, 2, 2).has_value());
}

TEST_CASE("Recording start times convert RF samples to MJD",
          "[stelline][fbh5_writer][module][timing]") {
    using Catch::Matchers::WithinAbs;

    CHECK(Modules::UnixSecondsToMjd(0.0L) == 40587.0);
    CHECK_THAT(Modules::UnixSecondsToMjd(Modules::RfSampleUnixSeconds(1700000000, 48, 1e-3)),
               WithinAbs(40587.0 + (1700000000.0 + 0.048) / 86400.0, 1e-10));
}

TEST_CASE("Recording source checks reject any band or clock change",
          "[stelline][fbh5_writer][module][metadata]") {
    struct Text {
        std::string value;
        JST_SERDES(value);
    };
    struct Real {
        F64 value = 0.0;
        JST_SERDES(value);
    };
    struct Integer {
        I64 value = 0;
        JST_SERDES(value);
    };

    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    auto& environment = flowgraph.environment();
    const std::string tuning = "observatory.antenna.1a.tunings.a";
    const auto setAll = [&]() {
        REQUIRE(environment.set("instance.bands.len", Integer{1}) == Result::SUCCESS);
        REQUIRE(environment.set("observation.antennas.0", Text{"1a"}) == Result::SUCCESS);
        REQUIRE(environment.set("instance.bands.0.tuning", Text{"a"}) == Result::SUCCESS);
        REQUIRE(environment.set("instance.bands.0.band_index", Integer{0}) == Result::SUCCESS);
        REQUIRE(environment.set(tuning + ".bands.0.frequency_start", Real{1000.0}) == Result::SUCCESS);
        REQUIRE(environment.set(tuning + ".bands.0.frequency_stop", Real{1002.0}) == Result::SUCCESS);
        REQUIRE(environment.set(tuning + ".bands.0.channel_start", Integer{352}) == Result::SUCCESS);
        REQUIRE(environment.set(tuning + ".bands.0.channel_stop", Integer{354}) == Result::SUCCESS);
        REQUIRE(environment.set(tuning + ".fengine.synctime", Integer{1700000000}) == Result::SUCCESS);
        REQUIRE(environment.set(tuning + ".fengine.sample_period", Real{1e-3}) == Result::SUCCESS);
    };

    setAll();
    Modules::RecordingSource expected;
    REQUIRE(Modules::LoadRecordingSource(environment, "[TEST]", expected) == Result::SUCCESS);
    CHECK(expected.channelCount() == 2);

    REQUIRE(environment.set("instance.bands.len", Integer{2}) == Result::SUCCESS);
    Modules::RecordingSource multiple;
    CHECK(Modules::LoadRecordingSource(environment, "[TEST]", multiple) == Result::ERROR);
    setAll();
    REQUIRE(Modules::CheckRecordingSource(environment, "[TEST]", expected) == Result::SUCCESS);

    const std::vector<std::pair<std::string, std::function<Result()>>> changes = {
        {"frequency_start", [&] { return environment.set(tuning + ".bands.0.frequency_start", Real{1000.5}); }},
        {"frequency_stop", [&] { return environment.set(tuning + ".bands.0.frequency_stop", Real{1003.0}); }},
        {"channel_start", [&] { return environment.set(tuning + ".bands.0.channel_start", Integer{350}); }},
        {"channel_stop", [&] { return environment.set(tuning + ".bands.0.channel_stop", Integer{356}); }},
        {"synctime", [&] { return environment.set(tuning + ".fengine.synctime", Integer{1700000001}); }},
        {"sample_period", [&] { return environment.set(tuning + ".fengine.sample_period", Real{2e-3}); }},
        {"band_index", [&] { return environment.set("instance.bands.0.band_index", Integer{1}); }},
        {"band_count", [&] { return environment.set("instance.bands.len", Integer{2}); }},
    };
    for (const auto& [name, change] : changes) {
        INFO(name);
        REQUIRE(change() == Result::SUCCESS);
        CHECK(Modules::CheckRecordingSource(environment, "[TEST]", expected) == Result::ERROR);
        setAll();
        REQUIRE(Modules::CheckRecordingSource(environment, "[TEST]", expected) == Result::SUCCESS);
    }

    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}

namespace {

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream), {});
}

}  // namespace

TEST_CASE("Recording staging files never reuse an existing path",
          "[stelline][fbh5_writer][module][lifecycle]") {
    const ScratchDirectory scratch;
    const auto& directory = scratch.path;
    const auto path = directory / "scan.fbh5";
    std::ofstream(path.string() + ".partial") << "foreign";

    std::filesystem::path first;
    std::filesystem::path second;
    REQUIRE_FALSE(Modules::ReservePartialRecording(path, first));
    REQUIRE_FALSE(Modules::ReservePartialRecording(path, second));
    CHECK(first != second);
    CHECK(first.parent_path() == directory);
    CHECK(std::filesystem::is_regular_file(first));
    CHECK(std::filesystem::is_regular_file(second));

    Modules::RemovePartialRecording(first);
    Modules::RemovePartialRecording(second);
    CHECK(first.empty());
    CHECK(ReadText(path.string() + ".partial") == "foreign");
    CHECK(std::distance(std::filesystem::directory_iterator(directory), {}) == 1);
}

TEST_CASE("Recording publication honors overwrite at the moment of publication",
          "[stelline][fbh5_writer][module][lifecycle]") {
    const ScratchDirectory scratch;
    const auto& directory = scratch.path;
    const auto path = directory / "scan.fbh5";

    std::filesystem::path partial;
    REQUIRE_FALSE(Modules::ReservePartialRecording(path, partial));
    std::ofstream(partial) << "new";
    std::ofstream(path) << "appeared";

    const auto refused = Modules::PublishPartialRecording(partial, path, false);
    CHECK(refused == std::errc::file_exists);
    CHECK(ReadText(path) == "appeared");
    REQUIRE_FALSE(partial.empty());
    CHECK(ReadText(partial) == "new");

    REQUIRE_FALSE(Modules::PublishPartialRecording(partial, path, true));
    CHECK(partial.empty());
    CHECK(ReadText(path) == "new");
    CHECK(std::distance(std::filesystem::directory_iterator(directory), {}) == 1);

    std::filesystem::remove(path);
    REQUIRE_FALSE(Modules::ReservePartialRecording(path, partial));
    std::ofstream(partial) << "fresh";
    REQUIRE_FALSE(Modules::PublishPartialRecording(partial, path, false));
    CHECK(ReadText(path) == "fresh");
    CHECK(std::distance(std::filesystem::directory_iterator(directory), {}) == 1);
}
