#include <stelline/domains/transport/dummy_receiver/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <stelline/domains/transport/dummy_receiver/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct DummyReceiverImpl : public Block::Impl, public DynamicConfig<Blocks::DummyReceiver> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::DummyReceiver> moduleConfig = std::make_shared<Modules::DummyReceiver>();
    Modules::DummyReceiverImpl* moduleImpl = nullptr;
};

Result DummyReceiverImpl::configure() {
    moduleConfig->shape = shape;
    moduleConfig->dataType = dataType;
    moduleConfig->period = period;

    return Result::SUCCESS;
}

Result DummyReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("output",
                                    "Output",
                                    "Generated tensor with fake data."));

    JST_CHECK(defineInterfaceConfig("shape",
                                    "Shape",
                                    "Output tensor shape as [antennas, channels, samples, polarizations].",
                                    "vector:int:dim"));

    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "Data type for generated samples.",
                                    "dropdown:CF32(CF32),F32(F32)"));

    JST_CHECK(defineInterfaceConfig("period",
                                    "Period",
                                    "Time between blocks in milliseconds.",
                                    "int:ms"));

    JST_CHECK(defineInterfaceMetric("timestamp",
                                    "Timestamp",
                                    "Current block timestamp.",
                                    "stelline-metrics",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getTimestamp() : U64(0)));
        }));

    JST_CHECK(defineInterfaceMetric("timestampDisplay",
                                    "Latest Timestamp",
                                    "Current block timestamp.",
                                    "label",
        [this]() -> std::any {
            return std::any(jst::fmt::format("{}", moduleImpl ? moduleImpl->getTimestamp() : U64(0)));
        }));

    return Result::SUCCESS;
}

Result DummyReceiverImpl::create() {
    JST_CHECK(moduleCreate("dummy_receiver", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("output", {"dummy_receiver", "output"}));

    moduleImpl = moduleHandle("dummy_receiver")->getImpl<Modules::DummyReceiverImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DummyReceiverImpl);

}  // namespace Jetstream::Blocks
