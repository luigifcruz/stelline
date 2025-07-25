#ifndef STELLINE_BITS_SOCKET_BASE_HH
#define STELLINE_BITS_SOCKET_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/socket/base.hh>

namespace stelline::bits::socket {

inline BitInterface SocketBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::socket;

    // Create metadata storage.

    auto metadata = std::make_shared<MetadataStorage>();

    // Fetch configuration YAML.

    auto mode = FetchNodeArg<std::string>(app, config, "mode");
    auto address = FetchNodeArg<std::string>(app, config, "address", "tcp://*:5555");

    HOLOSCAN_LOG_INFO("Socket Configuration:");
    HOLOSCAN_LOG_INFO("  Mode: {}", mode);
    HOLOSCAN_LOG_INFO("  Address: {}", address);

    // Declare modes.

    auto zmq_transmitter_op_cb = [&](const auto& op_id){
        return app->template make_operator<ZmqTransmitterOp>(
            op_id,
            Arg("address", address)
        );
    };

    // Select configuration mode.

    if (mode == "zmq") {
        HOLOSCAN_LOG_INFO("Creating ZeroMQ Transmitter operator.");
        const auto& op_id = fmt::format("socket-zmq-transmitter-{}", id);
        auto zmq_op = zmq_transmitter_op_cb(op_id);
        zmq_op->load_metadata(op_id, metadata);
        return {zmq_op, zmq_op, zmq_op};
    }

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::socket

#endif  // STELLINE_BITS_SOCKET_BASE_HH
