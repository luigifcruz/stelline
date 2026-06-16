#include <stelline/dummy_writer/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/dummy_writer/module.hh>

#include "module_impl.hh"

namespace Jetstream::Blocks {

struct DummyWriterImpl : public Block::Impl, public DynamicConfig<Blocks::DummyWriter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::DummyWriter> moduleConfig = std::make_shared<Modules::DummyWriter>();
    Modules::DummyWriterImpl* moduleImpl = nullptr;
};

Result DummyWriterImpl::configure() {
    return Result::SUCCESS;
}

Result DummyWriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input",
                                   "Input",
                                   "Input tensor to receive."));

    JST_CHECK(defineInterfaceMetric("iterations",
                                    "Iterations",
                                    "Total number of received blocks.",
                                    "private-stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getIterationCount() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("averageDurationMs",
                                    "Average Duration",
                                    "Average duration in milliseconds per received block.",
                                    "private-stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f}", moduleImpl ? moduleImpl->getAverageDurationMs() : F64(0.0)));
        }));

    JST_CHECK(defineInterfaceMetric("latestTimestamp",
                                    "Latest Timestamp",
                                    "Latest received timestamp.",
                                    "private-stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getLatestTimestamp() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("latestTimestampDisplay",
                                    "Latest Timestamp",
                                    "Latest received timestamp.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getLatestTimestamp() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("blocksProcessedDisplay",
                                    "Blocks Processed",
                                    "Total blocks processed.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getIterationCount() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("msPerBlockDisplay",
                                    "Time Per Block",
                                    "Average milliseconds per block.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{:.2f}ms", moduleImpl ? moduleImpl->getAverageDurationMs() : F64(0.0)));
        }));

    return Result::SUCCESS;
}

Result DummyWriterImpl::create() {
    JST_CHECK(moduleCreate("dummy_writer", moduleConfig, {
        {"input", inputs().at("input")}
    }));

    moduleImpl = moduleHandle("dummy_writer")->getImpl<Modules::DummyWriterImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DummyWriterImpl);

}  // namespace Jetstream::Blocks
