#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>

#include <jetstream/memory/axis.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>

#include <stelline/fbh5_writer/module.hh>

#include "module_impl.hh"

using namespace Jetstream;

namespace {

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
        return timeline.plan(input, gap);
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
    CHECK(timeline.plan(input, gap) == Result::ERROR);
    REQUIRE(plan(8232, 8) == Result::SUCCESS);
    CHECK(gap.missingBlocks == 0);

    Modules::RecordingTimeline untimed;
    REQUIRE(input.removeAttribute("duration") == Result::SUCCESS);
    REQUIRE(untimed.plan(input, gap) == Result::SUCCESS);
    CHECK_FALSE(gap.timing.has_value());
    untimed.commit(gap);
    REQUIRE(untimed.plan(input, gap) == Result::SUCCESS);
    REQUIRE(input.setAttribute("duration", U64{8}) == Result::SUCCESS);
    CHECK(untimed.plan(input, gap) == Result::ERROR);
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
