#include <stelline/domains/stelline/uvh5_writer/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/domains/stelline/uvh5_writer/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

struct Uvh5WriterImpl : public Block::Impl, public DynamicConfig<Blocks::Uvh5Writer> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Uvh5Writer> moduleConfig = std::make_shared<Modules::Uvh5Writer>();
    Modules::Uvh5WriterImpl* moduleImpl = nullptr;
};

Result Uvh5WriterImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->dspChannelizationRate = dspChannelizationRate;
    moduleConfig->dspIntegrationRate = dspIntegrationRate;
    moduleConfig->overwrite = overwrite;
    moduleConfig->recording = recording;

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input",
                                   "Input",
                                   "Pre-arranged rank-4 correlation tensor written to UVH5."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Destination UVH5 filepath.",
                                    "filesave:uvh5,h5"));

    JST_CHECK(defineInterfaceConfig("dspChannelizationRate",
                                    "DSP Channelization Rate",
                                    "DSP channelization factor used to derive the UVH5 frequency grid.",
                                    "int:"));

    JST_CHECK(defineInterfaceConfig("dspIntegrationRate",
                                    "DSP Integration Rate",
                                    "DSP integration factor used to derive the UVH5 integration time.",
                                    "int:"));

    JST_CHECK(defineInterfaceConfig("overwrite",
                                    "Overwrite",
                                    "Whether to overwrite the file if it already exists.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("recording",
                                    "Recording",
                                    "Start or stop recording to the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceMetric("bandwidth",
                                    "Bandwidth",
                                    "Write bandwidth in megabytes per second.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f}", moduleImpl ? moduleImpl->getBandwidthMBps() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("bandwidthDisplay",
                                    "Bandwidth",
                                    "Write bandwidth in megabytes per second.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f} MB/s", moduleImpl ? moduleImpl->getBandwidthMBps() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("totalDataWritten",
                                    "Total Data Written",
                                    "Total written UVH5 payload in megabytes.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.0f}", moduleImpl ? moduleImpl->getTotalDataWrittenMb() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("totalDataWrittenDisplay",
                                    "Total Data Written",
                                    "Total written UVH5 payload in megabytes.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.0f} MB", moduleImpl ? moduleImpl->getTotalDataWrittenMb() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("chunksWritten",
                                    "Chunks Written",
                                    "Total number of written UVH5 chunks.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getChunkCounter() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("chunksWrittenDisplay",
                                    "Chunks Written",
                                    "Total number of written UVH5 chunks.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getChunkCounter() : U64(0)));
        }));

    return Result::SUCCESS;
}

Result Uvh5WriterImpl::create() {
    JST_CHECK(moduleCreate("uvh5_writer", moduleConfig, {
        {"input", inputs().at("input")}
    }));

    moduleImpl = moduleHandle("uvh5_writer")->getImpl<Modules::Uvh5WriterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(Uvh5WriterImpl);

}  // namespace Jetstream::Blocks
