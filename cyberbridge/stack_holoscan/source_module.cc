
#include "cyberbridge/holoscan.hh"
#include "source_module.hh"

namespace Jetstream {

template<Device D, typename T>
void Source<D, T>::info() const {
    JST_DEBUG("  Node name: {}", config.nodeName);
    JST_DEBUG("  Node output name: {}", config.nodeOutputName);
}

template<Device D, typename T>
Result Source<D, T>::create() {
    JST_DEBUG("[BRIDGE] Initializing Holoscan Source module.");
    JST_INIT_IO();

    // Validate config values.

    if (config.nodeName.empty()) {
        JST_ERROR("Node name is empty.");
        return Result::ERROR;
    }

    if (config.nodeOutputName.empty()) {
        JST_ERROR("Node output name is empty.");
        return Result::ERROR;
    }

    // Check if the app is running.

    if (!CyberBridge::Holoscan::IsAppRunning()) {
        JST_ERROR("Holoscan app is not running.");
        return Result::ERROR;
    }

    // Register the tap.

    holoscanModuleId = jst::fmt::format("cyberbridge-{}", locale());

    JST_CHECK(CyberBridge::Holoscan::RegisterTap(holoscanModuleId, [&](){
        auto& app = CyberBridge::Holoscan::GetApp();

        // Check if the node exists.

        const auto& node = app.graph().find_node(config.nodeName);
        if (!node) {
            JST_ERROR("Node '{}' does not exist.", config.nodeName);
            return Result::ERROR;
        }

        // Check if the node output exists.

        if (!node->spec()->outputs().contains(config.nodeOutputName)) {
            JST_ERROR("Node output '{}' does not exist.", config.nodeOutputName);
            return Result::ERROR;
        }

        // NOTE: CyberBridge will assume the input data is composed of Jetstream::Tensor
        //       objects. Other data types can be supported in the future without too much
        //       effort. For now, this is enough.

        // Check node output data type and device.

        const auto& spec = node->spec()->outputs().at(config.nodeOutputName);
        if (*spec->typeinfo() != typeid(std::shared_ptr<Tensor<D, T>>)) {
            JST_ERROR("Node output '{}' is not of type Tensor<{}, {}>.", config.nodeOutputName,
                                                                            GetDevicePrettyName(D),
                                                                            NumericTypeInfo<T>::name);
            return Result::ERROR;
        }

        // Create the operator.

        const auto& tx = app.graph().find_node(config.nodeName);
        const auto& txName = holoscanModuleId.c_str();
        op = app.make_operator<Op>(txName);
        op->shape() = config.shape;

        app.add_flow(tx, op, {{config.nodeOutputName, "in"}});

        return Result::SUCCESS;
    }));

    // Allocate the output buffer.

    output.buffer = Tensor<D, T>(config.shape);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Source<D, T>::destroy() {
    JST_DEBUG("[BRIDGE] Destroying Holoscan Source module.");

    JST_CHECK(CyberBridge::Holoscan::RemoveTap(holoscanModuleId));

    return Result::SUCCESS;
}

template<Device D, typename T>
const std::string& Source<D, T>::warning() const {
    return _warning;
}

template<Device D, typename T>
Result Source<D, T>::computeReady() {
    if (!op->buffer.wait()) {
        return Result::TIMEOUT;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Source<D, T>::compute(const Context&) {
    if (!_warning.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return Result::SUCCESS;
    }

    // Running inside a locked context because the operator can 
    // be reloaded at any time by the bridge.
    return CyberBridge::Holoscan::LockContext([&]{
        Tensor<D, T> tensor;

        if (op->status().has_value()) {
            _warning = jst::fmt::format("Holoscan operator failed: {}", op->status().value());
            return Result::SKIP;
        }

        if (!op->buffer.get(tensor)) {
            return Result::SKIP;
        }

        JST_CHECK(Memory::Copy(output.buffer, tensor));

        return Result::SUCCESS;
    });
}

//
// Holoscan Operator
//

template<Device D, typename T>
void Source<D, T>::Op::setup(holoscan::OperatorSpec& spec) { 
    spec.input<std::shared_ptr<Tensor<D, T>>>("in"); 
}

template<Device D, typename T>
void Source<D, T>::Op::start() {
    buffer.resize(5, _shape);
    _status.reset();
}

template<Device D, typename T>
void Source<D, T>::Op::compute(holoscan::InputContext& input, 
                               holoscan::OutputContext&, 
                               holoscan::ExecutionContext&) {
    auto in_value = input.receive<std::shared_ptr<Tensor<D, T>>>("in").value();

    buffer.put([&](auto& tensor){
        if (in_value->shape() != tensor.shape()) {
            _status = jst::fmt::format("Tensor shape {} does not match the "
                                       "configured buffer shape {}.", in_value->shape(), 
                                                                      tensor.shape());
            return;
        }

        // TODO: Replace with non-throwing check.
        JST_CHECK_THROW(Memory::Copy(tensor, *in_value));
    });
};

template<Device D, typename T>
const std::optional<std::string>& Source<D, T>::Op::status() const {
    return _status;
}

template<Device D, typename T>
std::vector<U64>& Source<D, T>::Op::shape() {
    return _shape;
}

JST_SOURCE_CPU(JST_INSTANTIATION)
JST_SOURCE_CUDA(JST_INSTANTIATION)

}  // namespace Jetstream