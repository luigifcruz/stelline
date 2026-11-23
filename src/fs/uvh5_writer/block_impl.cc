#include <stelline/uvh5_writer/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/uvh5_writer/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>
#include <jetstream/domains/core/permutation/module.hh>

#include "module_impl.hh"
#include "../recording_input.hh"

namespace Jetstream::Blocks {

struct Uvh5WriterImpl : public Block::Impl, public DynamicConfig<Blocks::Uvh5Writer> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Uvh5Writer> moduleConfig = std::make_shared<Modules::Uvh5Writer>();
    std::shared_ptr<Modules::Permutation> permutationConfig = std::make_shared<Modules::Permutation>();
    std::shared_ptr<Modules::Duplicate> duplicateConfig = std::make_shared<Modules::Duplicate>();
    Modules::Uvh5WriterImpl* moduleImpl = nullptr;
};

Result Uvh5WriterImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->overwrite = overwrite;
    moduleConfig->recording = recording;

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input",
                                   "Input",
                                   "Rank-4 CF32 correlation tensor written to UVH5."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Destination UVH5 filepath.",
                                    {{"type", "filesave"}, {"extensions", Parser::MakeSequence({"uvh5", "h5"})}}));

    JST_CHECK(defineInterfaceConfig("overwrite",
                                    "Overwrite",
                                    "Whether to overwrite the file if it already exists.",
                                    {{"type", "bool"}}));

    JST_CHECK(defineInterfaceConfig("recording",
                                    "Recording",
                                    "Start or stop recording to the file.",
                                    {{"type", "bool"}}));

    JST_CHECK(defineInterfaceMetric("bandwidth",
                                    "Bandwidth",
                                    "Write bandwidth in megabytes per second.",
                                    {{"type", "stelline-metrics-number"}, {"visibility", "internal"}},
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f}", moduleImpl ? moduleImpl->getBandwidthMBps() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("bandwidthDisplay",
                                    "Bandwidth",
                                    "Write bandwidth in megabytes per second.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f} MB/s", moduleImpl ? moduleImpl->getBandwidthMBps() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("totalDataWritten",
                                    "Total Data Written",
                                    "Total written UVH5 payload in megabytes.",
                                    {{"type", "stelline-metrics-number"}, {"visibility", "internal"}},
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.0f}", moduleImpl ? moduleImpl->getTotalDataWrittenMb() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("totalDataWrittenDisplay",
                                    "Total Data Written",
                                    "Total written UVH5 payload in megabytes.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.0f} MB", moduleImpl ? moduleImpl->getTotalDataWrittenMb() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("chunksWritten",
                                    "Chunks Written",
                                    "Total number of written UVH5 chunks.",
                                    {{"type", "stelline-metrics-number"}, {"visibility", "internal"}},
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getChunkCounter() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("chunksWrittenDisplay",
                                    "Chunks Written",
                                    "Total number of written UVH5 chunks.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getChunkCounter() : U64(0)));
        }));

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::create() {
    TensorLink input = inputs().at("input");
    Modules::RecordingInputPlan plan;
    if (device() == DeviceType::CUDA) {
        JST_CHECK(Modules::PlanRecordingInput(input.tensor, "[BLOCK_UVH5_WRITER]", plan));
    }
    if (plan.permutation) {
        permutationConfig->permutation = *plan.permutation;
        JST_CHECK(moduleCreate("permutation", permutationConfig, {
            {"buffer", input}
        }));
        input = moduleGetOutput({"permutation", "buffer"});
    }
    if (plan.copy) {
        duplicateConfig->outputDevice = "cuda";
        duplicateConfig->hostAccessible = false;
        JST_CHECK(moduleCreate("duplicate", duplicateConfig, {
            {"buffer", input}
        }));
        input = moduleGetOutput({"duplicate", "buffer"});
    }

    JST_CHECK(moduleCreate("uvh5_writer", moduleConfig, {
        {"input", input}
    }));

    moduleImpl = moduleHandle("uvh5_writer")->getImpl<Modules::Uvh5WriterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(Uvh5WriterImpl, {"stelline_uvh5_writer"});

}  // namespace Jetstream::Blocks
