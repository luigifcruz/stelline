#include <stelline/fbh5_reader/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/fbh5_reader/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

struct Fbh5ReaderImpl : public Block::Impl, public DynamicConfig<Blocks::Fbh5Reader> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Fbh5Reader> moduleConfig = std::make_shared<Modules::Fbh5Reader>();
    Modules::Fbh5ReaderImpl* moduleImpl = nullptr;
};

Result Fbh5ReaderImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->batchSize = batchSize;
    moduleConfig->loop = loop;
    moduleConfig->playing = playing;

    return Result::SUCCESS;
}

Result Fbh5ReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing filterbank data read from the file."));

    JST_CHECK(defineInterfaceOutput("mask",
                                    "Mask",
                                    "The output mask as read from the file."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Destination FBH5 filepath.",
                                    "filepicker:fbh5,h5"));

    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Number of time-indices to read per processing cycle.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("loop",
                                    "Loop",
                                    "Whether to loop back to the start when reaching the end of the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("playing",
                                    "Playing",
                                    "Start or stop reading from the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current file position.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const U64 count = moduleImpl->getBatchCount();
            if (count == 0) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 progress = static_cast<F32>(moduleImpl->getCurrentBatchIndex()) /
                                 static_cast<F32>(count);
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", progress * 100.0f), progress};
        }));

    JST_CHECK(defineInterfaceMetric("currentBandwidth",
                                    "Bandwidth",
                                    "Smoothed recent file read rate.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s", moduleImpl->getCurrentBandwidth());
        }));

    return Result::SUCCESS;
}

Result Fbh5ReaderImpl::create() {
    JST_CHECK(moduleCreate("fbh5_reader", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"fbh5_reader", "signal"}));
    JST_CHECK(moduleExposeOutput("mask", {"fbh5_reader", "mask"}));

    moduleImpl = moduleHandle("fbh5_reader")->getImpl<Modules::Fbh5ReaderImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(Fbh5ReaderImpl);

}  // namespace Jetstream::Blocks
