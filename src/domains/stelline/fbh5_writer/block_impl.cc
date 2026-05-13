#include <stelline/domains/stelline/fbh5_writer/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/domains/stelline/fbh5_writer/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

struct Fbh5WriterImpl : public Block::Impl, public DynamicConfig<Blocks::Fbh5Writer> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Fbh5Writer> moduleConfig = std::make_shared<Modules::Fbh5Writer>();
    Modules::Fbh5WriterImpl* moduleImpl = nullptr;
};

Result Fbh5WriterImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->overwrite = overwrite;
    moduleConfig->recording = recording;

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input",
                                   "Input",
                                   "Pre-arranged rank-4 tensor written to FBH5."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Destination FBH5 filepath.",
                                    "filesave:fbh5,h5"));

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
                                    "private-stelline-metrics",
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
                                    "Total written FBH5 payload in megabytes.",
                                    "private-stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.0f}", moduleImpl ? moduleImpl->getTotalDataWrittenMb() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("totalDataWrittenDisplay",
                                    "Total Data Written",
                                    "Total written FBH5 payload in megabytes.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.0f} MB", moduleImpl ? moduleImpl->getTotalDataWrittenMb() : 0.0));
        }));

    JST_CHECK(defineInterfaceMetric("chunksWritten",
                                    "Chunks Written",
                                    "Total number of written FBH5 chunks.",
                                    "private-stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getChunkCounter() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("chunksWrittenDisplay",
                                    "Chunks Written",
                                    "Total number of written FBH5 chunks.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getChunkCounter() : U64(0)));
        }));

    return Result::SUCCESS;
}

Result Fbh5WriterImpl::create() {
    JST_CHECK(moduleCreate("fbh5_writer", moduleConfig, {
        {"input", inputs().at("input")}
    }));

    moduleImpl = moduleHandle("fbh5_writer")->getImpl<Modules::Fbh5WriterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(Fbh5WriterImpl);

}  // namespace Jetstream::Blocks
