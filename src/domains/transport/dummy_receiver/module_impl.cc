#include "module_impl.hh"

namespace Jetstream::Modules {

Result DummyReceiverImpl::validate() {
    const auto& config = *candidate();

    if (config.shape.size() != 4) {
        JST_ERROR("[MODULE_DUMMY_RECEIVER] Shape must have 4 dimensions [A, C, S, P].");
        return Result::ERROR;
    }

    for (const auto& dim : config.shape) {
        if (dim == 0) {
            JST_ERROR("[MODULE_DUMMY_RECEIVER] Shape dimensions must be positive.");
            return Result::ERROR;
        }
    }

    if (config.dataType != "CF32" && config.dataType != "F32") {
        JST_ERROR("[MODULE_DUMMY_RECEIVER] Unsupported data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DummyReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("output"));

    return Result::SUCCESS;
}

Result DummyReceiverImpl::create() {
    const auto& config = *candidate();

    period = config.period;

    JST_CHECK(outputTensor.create(device(), NameToDataType(config.dataType), config.shape));

    outputTensor.setAttribute("timestamp", timestamp);

    outputs()["output"].produced(name(), "output", outputTensor);

    return Result::SUCCESS;
}

Result DummyReceiverImpl::destroy() {
    return Result::SUCCESS;
}

const U64& DummyReceiverImpl::getTimestamp() const {
    return timestamp;
}

const U64& DummyReceiverImpl::getPeriod() const {
    return period;
}

}  // namespace Jetstream::Modules
