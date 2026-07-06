#include <stelline/uvh5_reader/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/uvh5_reader/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

struct Uvh5ReaderImpl : public Block::Impl, public DynamicConfig<Blocks::Uvh5Reader> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Uvh5Reader> moduleConfig = std::make_shared<Modules::Uvh5Reader>();
    Modules::Uvh5ReaderImpl* moduleImpl = nullptr;
};

Result Uvh5ReaderImpl::configure() {
    moduleConfig->filepath = filepath;
    moduleConfig->batchSize = batchSize;
    moduleConfig->loop = loop;
    moduleConfig->playing = playing;

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "CF32 tensor [batchSize, Nspws x Nfreqs, Npols] of visibilities."));

    JST_CHECK(defineInterfaceConfig("filepath",
                                    "File Path",
                                    "Path to the UVH5 (HDF5) correlator visibility file.",
                                    "filepicker:uvh5,h5,hdf5"));

    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Number of baseline-time (blt) entries to read per cycle.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("loop",
                                    "Loop",
                                    "Restart from the beginning when reaching the end of the file.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("playing",
                                    "Playing",
                                    "Pause or resume reading.",
                                    "bool"));

    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current position in the file.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const U64 total = moduleImpl->getTotalBlts();
            if (total == 0) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 progress = static_cast<F32>(moduleImpl->getCurrentBlt()) /
                                 static_cast<F32>(total);
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", progress * 100.0f), progress};
        }));

    JST_CHECK(defineInterfaceMetric("currentBandwidth",
                                    "Bandwidth",
                                    "Smoothed HDF5 read throughput.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return jst::fmt::format("{:.1f} MB/s", moduleImpl->getCurrentBandwidth());
        }));

    JST_CHECK(defineInterfaceMetric("fileInfo",
                                    "File Info",
                                    "Number of baseline-time entries in the opened dataset.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("-");
            }
            const U64 total = moduleImpl->getTotalBlts();
            if (total == 0) {
                return std::string("-");
            }
            return jst::fmt::format("{} blts", total);
        }));

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::create() {
    JST_CHECK(moduleCreate("uvh5_reader", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"uvh5_reader", "signal"}));

    moduleImpl = moduleHandle("uvh5_reader")->getImpl<Modules::Uvh5ReaderImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(Uvh5ReaderImpl);

}  // namespace Jetstream::Blocks
